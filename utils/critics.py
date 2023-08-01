#from Aurora.agent.geometric.util import batch_to_gd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List
from torch_geometric.nn import RGCNConv, MessagePassing
from torch_geometric.data import Data as GeometricData, Batch
from gym_minigrid.minigrid import DIR_TO_VEC


class RelationalCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """
    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self,
                 # sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                  #obj_n: int,
                 n_agents: int,
                 spatial_tensors,
                 batch_size: int,
                 n_actions: int,
                 input_dims: list,
                 hidden_dim: int = 32,
                 relational_embedding : bool = False,
                 norm_in: object = True,
                 net_code: object = "1g0f",
                 mp_rounds: object = 1) -> object:
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.max_reduce = True # TODO hardcoded
        self.relational_embedding = relational_embedding

        self.spatial_tensors = np.array(spatial_tensors)
        self.binary_batch = torch.tensor([self.spatial_tensors for _ in range(self.batch_size)])
        self.gd, self.slices = batch_to_gd(self.binary_batch, self.device)  # makes adjs geometric data usable for torch geometric
        self.nb_edge_types = len(spatial_tensors)

        # hidden_dim = 14
        self.rel_embedder = nn.Linear(14, hidden_dim) # TODO hardcoded
        self.embedder = nn.Linear(input_dims[0], hidden_dim)
        if not self.relational_embedding:
            self.gnn_layers = RGCNConv(hidden_dim, hidden_dim, self.nb_edge_types)
        else:
            print(f'We are using relational embeddings')
            self.gnn_layers = MPLayer(hidden_dim)
        # iterate over agents
        for _ in range(self.n_agents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim + self.num_actions * (self.n_agents-1),
                                                      hidden_dim)) # takes in 128+6*2 , out 128

            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent

        self.shared_modules = [self.embedder, self.gnn_layers]


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
            p.grad.data.mul_(1. / self.n_agents)

    def forward(self,
                obs,
                unary_tensors,
                actions,
                binary_tensors = None,
                agents=None,
                return_q=True,
                return_all_q=False,
                regularize = False,
                logger=None,
                niter=0):
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
        if agents is None:
            agents = range(self.n_agents)

        all_rets = []

        for a_i in agents:
            # feature embedding
            embedds = torch.flatten(unary_tensors[a_i], 0, 1).float().to(device=self.device)
            embedds = self.embedder(embedds) # seems right so far

            # RGCN module
            if all(binary_tensors):
                gd = Batch.from_data_list(binary_tensors[a_i])
                gd = gd.to(device = self.device)
                rel = self.rel_embedder(gd.edge_attr)
                embedds = self.gnn_layers(embedds, gd.edge_index, rel)
            else:
                embedds = self.gnn_layers(embedds, self.gd.edge_index, self.gd.edge_attr)
            embedds = torch.relu(embedds)

            chunks = torch.split(embedds, self.slices, dim=0) # splits it in slices/entities
            chunks = [p.unsqueeze(0) for p in chunks] # just adds back another dimension in the beginning
            x = torch.cat(chunks, dim=0)
            if self.max_reduce:
                # max-pooling layer
                x, _ = torch.max(x, dim=1) # TODO check what dimension comes out of here
            else:
                # I think this would be for the CNN which is flattened
                x = torch.flatten(x, start_dim=1, end_dim=2) # TODO what does that do?

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
    return geometric_batch.to(device=device), slices

class MPLayer(MessagePassing):
    def __init__(self, in_channels):
        """
        in_channels: embedding size for nodes and edges
        """
        super(MPLayer, self).__init__(aggr="add")

        self.message_mlp = nn.Linear(in_channels * 3, in_channels) # 3 because we have two node embeddings and one edge embedding feeding in
        self.node_update = nn.Linear(in_channels * 2, in_channels) # 1 previous embedding and one aggregated message

    def message(self, x_j, x_i, edge_attr):
        """
        Constructs a message to node i got each edge (by default from j to i) given the edge attributes and node embeddings
        """
        x = torch.cat((x_i, x_j, edge_attr), dim=1)
        return torch.relu(self.message_mlp(x))

    def update(self, aggr_out, x):
        '''
        Updates node embedding.
        Parameters:
            aggr_out: output of aggregation
            x: any other argument that was initially passed to propagate, so I believe the node feauture?
        Returns:
            Updated node embedding
        '''
        x = torch.cat((x, aggr_out), dim=1)
        return torch.relu(self.node_update(x))

    def forward(self, node_features, edge_index, edge_attr):
        """
        Takes in x=node_features [N, in_channels], edge_index [2, E], edge_attr [E, in_channels ]
        Parameters:
            node_features: also denotes as x [N, in_channels] with N being the number of nodes
        """
        return self.propagate(edge_index=edge_index, x=node_features, edge_attr=edge_attr)
