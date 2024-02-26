import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from torch_geometric.nn import RGCNConv, pool, RGATConv, GATv2Conv, GATConv
from torch_geometric.data import Data as GeometricData, Batch

class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x, batch):
        return pool.global_max_pool(x, batch)


class RelationalCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """
    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self,
                 n_agents: int,
                 spatial_tensors,
                 batch_size: int,
                 n_actions: int,
                 input_dims: list,
                 dense: bool,
                 hidden_dim: int = 32,
                 device: str = 'cuda:0',
                 graph_layer: str = 'RGCN',
                 ) -> object:
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """
        super(RelationalCritic, self).__init__()
        # self.sa_sizes = sa_sizes # TODO sa_sizes not needed in here anymore
        self.n_agents = n_agents
        self.num_actions = n_actions[0]
        self.critics_head = nn.ModuleList()
        self.device = device
        self.batch_size = batch_size
        self.max_reduce = True # TODO hardcoded
        self.dense = dense
        self.graph_layer = graph_layer
        self.spatial_tensors = np.array(spatial_tensors)
        self.binary_batch = torch.tensor([self.spatial_tensors for _ in range(self.batch_size)])
        self.gd, self.slices = batch_to_gd(self.binary_batch, self.device)  # makes adjs geometric data usable for torch geometric
        self.nb_edge_types = len(spatial_tensors)

        # self.embedder = nn.Linear(input_dims[0], hidden_dim)
        if self.graph_layer == 'RGCN':
            self.gnn_layers = RGCNConv(in_channels=input_dims[0],
                                       out_channels = hidden_dim,
                                       num_relations = self.nb_edge_types
                                       )
        elif self.graph_layer == 'RGAT':
            self.gnn_layers = RGATConv(in_channels=input_dims[0],
                                       out_channels=hidden_dim,
                                       num_relations = self.nb_edge_types
                                       )
            print('This is using RGAT as graph layer')
        elif self.graph_layer == 'GAT':
            attend_heads = 4
            assert (hidden_dim % attend_heads) == 0
            attend_dim = hidden_dim // attend_heads

            self.gnn_layers = GATConv(in_channels = input_dims[0],
                                      out_channels=attend_dim,
                                      heads=attend_heads,
                                      v2 = True
                                      )
        elif self.graph_layer == 'GATv2':
            attend_heads = 4
            assert (hidden_dim % attend_heads) == 0
            attend_dim = hidden_dim // attend_heads

            self.gnn_layers = GATv2Conv(in_channels = input_dims[0],
                                      out_channels=attend_dim,
                                      heads=attend_heads,
                                      )
            print('Using GATv2 layer')
        else:
            print('not a valid graph layer')
        print(f'Using {self.graph_layer} as graph layer')
        # iterate over agents
        self.pooling = Pool()
        for _ in range(self.n_agents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim + self.num_actions * (self.n_agents-1),
                                                      hidden_dim)) # takes in 128+6*2 , out 128

            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent

        # self.shared_modules = [self.embedder, self.gnn_layers]
        self.shared_modules = [self.gnn_layers]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            if p.grad is not None: 
                p.grad.data.mul_(1. / self.n_agents)

    def forward(self,
                obs,
                unary_tensors,
                actions,
                binary_tensors = None,
                agents=None,
                return_q=True,
                return_all_q=False,):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        self.node_embeddings = []
        self.node_concepts = []
        self.graph_embeddings = []
        if agents is None:
            agents = range(self.n_agents)

        all_rets = []

        for a_i in agents:

            if self.dense:
                gd = Batch.from_data_list(binary_tensors[a_i])
                gd = gd.to(device = self.device)
                if 'GAT' in self.graph_layer:
                    embedds = self.gnn_layers(gd.x, gd.edge_index)
                else:
                    embedds = self.gnn_layers(gd.x, gd.edge_index, gd.edge_attr)
                self.node_embeddings.append(embedds)
                embedds = torch.relu(embedds)
                self.node_concepts.append(embedds)
                x = pool.global_max_pool(embedds, gd.batch)
                self.graph_embeddings.append(x)
            else:
                embedds = torch.flatten(unary_tensors[a_i], 0, 1).float().to(device=self.device)
                embedds = self.gnn_layers(embedds, self.gd.edge_index, self.gd.edge_attr)
                embedds = torch.relu(embedds)
                x = self.pooling(embedds, self.gd.batch)


            # extract state encoding for each agent that we're returning Q for
            other_actions = actions.copy()
            other_actions.pop(a_i)
            critic_in = torch.cat((x, *other_actions), dim=1)
            agent_rets = []
            all_q = self.critics_head[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        return all_rets


def batch_to_gd(batch: torch.Tensor, device: str):
    # [B x R x E x E]
    batch_size = batch.shape[0]
    nb_relations = batch.shape[1] #gets 14 out when full relations
    nb_objects = batch.shape[2] # gets n x n out

    assert batch.shape[2] == batch.shape[3]

    # index array for all entities
    x = torch.arange(nb_objects).view(-1, 1)

    # I guess this should split the batch into single chunks along the batch size
    i_lst = [x.view(nb_relations, nb_objects, nb_objects) for x in torch.split(batch, 1, dim=0)]

    def to_gd(tensor: torch.Tensor) -> GeometricData:
        """
        takes batch of adjacency geometric data and transforms it to a GeometricData object for torchgeometric
        """
        # tensor is one batch of multi-dimensional adjacency matrix
        nz = torch.nonzero(tensor)
        edge_attr = nz[:, 0]
        # edge_lst = nz[:, 1:].cpu().numpy().tolist()
        # edge_index = torch.LongTensor(list(zip(*edge_lst)))
        edge_index = nz[:, 1:].T
        return GeometricData(x=x, edge_index=edge_index, edge_attr=edge_attr)

    batch_data = [to_gd(instance) for instance in i_lst]

    geometric_batch = Batch.from_data_list(batch_data)
    max_node = max(i + 1 for b in batch_data for i in b.x[:, 0].cpu().numpy())
    slices = [max_node for _ in batch_data]
    # print(device)
    return geometric_batch.to(device=device), slices
