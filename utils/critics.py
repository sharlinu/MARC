#from Aurora.agent.geometric.util import batch_to_gd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List
from torch_geometric.nn import RGCNConv
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
                 rel_deter_func: list,
                 hidden_dim: int = 32,
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
        # self.dense = True
        self.rel_deter_func = rel_deter_func

        self.spatial_tensors = np.array(spatial_tensors)
        self.binary_batch = torch.tensor([self.spatial_tensors for _ in range(self.batch_size)])
        self.gd, self.slices = batch_to_gd(self.binary_batch, self.device)  # makes adjs geometric data usable for torch geometric
        self.nb_edge_types = len(spatial_tensors)


        self.embedder = nn.Linear(input_dims[0], hidden_dim) # TODO - hardcoding needed?
        self.gnn_layers = RGCNConv(hidden_dim, hidden_dim, self.nb_edge_types)

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
        # state = None
        # inputs = [[],
        #           torch.flatten(unary_tensor, 0, 1).float(), #flattens obs["unary_tensor"] only in 0th and 1st dim
        #           #torch.flatten(binary_tensor, 0, 1).permute(0,3,1,2).float()]
        #           binary_tensor.permute(0, 3, 1, 2).float()
        #           ]
        # for i in [1,2]:
        #     inputs[i] = inputs[i].to(device=device)
        # adj_matrices = inputs[2]
        all_rets = []

        for a_i in agents:
            # feature embedding
            #embedds = torch.flatten(inputs[1], 0, 1)
            embedds = torch.flatten(unary_tensors[a_i], 0, 1).float().to(device=self.device)
            #embedds = self.embedding_linear(embedds)
            embedds = self.embedder(embedds) # seems right so far


            # RGCN module
            if binary_tensors:

                # single_gd, self.slices = to_gd(binary_t)  # makes adjs geometric data usable for torch geometric
                # batch_data = [to_gd(instance) for instance in binary_tensors]

                gd = Batch.from_data_list(binary_tensors[a_i]).to(device = self.device)
                # max_node = max(i + 1 for b in batch_data for i in b.x[:, 0].cpu().numpy())
                # slices = [max_node for _ in batch_data]

                embedds = self.gnn_layers(embedds, gd.edge_index, gd.edge_attr)
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

    def preprocess(self, img_batch: np.array):
        spatial_tensors_list = []
        for pic in img_batch:
            pic = np.reshape(pic, (6,6,3)) #TODO hardcoded
            pic = pic.astype(np.int32) #1024, 108
        # data = filter_non_zero_elements(img) if self.dense else img
            objs = []
            for y, row in enumerate(pic):
                for x, pixel in enumerate(row):
                    # print(pixel)
                    if np.sum(pixel) == 0:
                        continue
                    obj = GridObject(x, y)
                    objs.append(obj)

            spatial_tensors = [np.zeros([len(objs), len(objs)]) for _ in
                               range(self.nb_edge_types)]
            for obj_idx1, obj1 in enumerate(objs):
                for obj_idx2, obj2 in enumerate(objs):
                    direction_vec = DIR_TO_VEC[1]
                    for rel_idx, func in enumerate(self.rel_deter_func):
                        if func(obj1, obj2, direction_vec):
                            spatial_tensors[rel_idx][obj_idx1, obj_idx2] = 1.0
            spatial_tensors_list.append(spatial_tensors)
        binary_batch = torch.tensor([spatial_tensors_list[i] for i in range(self.batch_size)])
        return binary_batch


def parse_code(net_code: str):
    """
    :param net_code: format <a>g[m]<b>f
    """
    assert net_code[1]=="g"
    assert net_code[-1]=="f"
    nb_gnn_layers = int(net_code[0])
    nb_dense_layers = int(net_code[-2])
    is_max = True if net_code[2] == "m" else False
    return nb_gnn_layers, nb_dense_layers, is_max



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


class GridObject:
    "object is specified by its location"
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # self.attributes = attributes
    @property
    def pos(self):
        return np.array([self.x, self.y])