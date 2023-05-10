#from Aurora.agent.geometric.util import batch_to_gd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data as GeometricData, Batch


# class TestCritic(nn.Module):
#     """
#     Relational network, used as critic for all agents.
#     """
#
#     # TODO previously embedding_size = 16, now we have hidden_dim and 32
#     def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
#                  # obj_n: int,
#                  n_actions: int,
#                  input_dims: list,
#                  hidden_dim: object = 32,
#                  norm_in: object = True,
#                  net_code: object = "1g0f",
#                  mp_rounds: object = 1) -> object:
#         """
#         Inputs:
#             sa_sizes (list of (int, int)): Size of state and action spaces per
#                                           agent
#             hidden_dim (int): Number of hidden dimensions
#             norm_in (bool): Whether to apply BatchNorm to input
#         """
#         super(TestCritic, self).__init__()
#         self.sa_sizes = sa_sizes
#         self.nagents = len(sa_sizes)
#         self.num_actions = n_actions[0]
#         self.critic_encoders = nn.ModuleList()
#         self.critics_head = nn.ModuleList()
#
#         # self.state_encoder = nn.ModuleList()
#         # iterate over agents
#         for _ in range(self.nagents):
#             critic = nn.Sequential()
#             critic.add_module('critic_fc1', nn.Linear(hidden_dim,  # critic only takes in 1* hidden_dim now
#                                                       hidden_dim))
#             critic.add_module('critic_nl', nn.LeakyReLU())
#             critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
#             self.critics_head.append(critic)  # one critic for each agent
#
#         self.embedder = nn.Linear(16, hidden_dim)  # TODO test
#
#         self.shared_modules = [self.embedder]
#         # self.shared_modules = [self.key_extractors, self.selector_extractors,
#         #                       self.value_extractors, self.critic_encoders]
#
#     def shared_parameters(self):
#         """
#         Parameters shared across agents and reward heads
#         """
#         return chain(*[m.parameters() for m in self.shared_modules])
#
#     def scale_shared_grads(self):
#         """
#         Scale gradients for parameters that are shared since they accumulate
#         gradients from the critic loss function multiple times
#         """
#         for p in self.shared_parameters():
#             p.grad.data.mul_(1. / self.nagents)
#
#     def forward(self,
#                 inps,
#                 agents=None,
#                 return_q=True,
#                 return_all_q=False,
#                 regularize=False,
#                 logger=None,
#                 niter=0):
#         """
#         Inputs:
#             inps (list of PyTorch Matrices): Inputs to each agents' encoder
#                                              (batch of obs + ac)
#             agents (int): indices of agents to return Q for
#             return_q (bool): return Q-value
#             return_all_q (bool): return Q-value for all actions
#             regularize (bool): returns values to add to loss function for
#                                regularization
#             return_attend (bool): return attention weights per agent
#             logger (TensorboardX SummaryWriter): If passed in, important values
#                                                  are logged
#         """
#         states = [s for s, a in inps]
#         actions = [a for s, a in inps]
#         # extract state-action encoding for each agent
#         s_encodings = self.embedder(states[0]) # TODO hardcoded
#
#         agents = range(1) # TODO hardcoded
#         all_rets = []
#         for i, a_i in enumerate(agents):
#             # extract state encoding for each agent that we're returning Q for
#             agent_rets = []
#             all_q = self.critics_head[a_i](s_encodings)
#             int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
#             q = all_q.gather(1, int_acs)
#             if return_q:
#                 agent_rets.append(q)
#             if return_all_q:
#                 agent_rets.append(all_q)
#             if len(agent_rets) == 1:
#                 all_rets.append(agent_rets[0])
#             else:
#                 all_rets.append(agent_rets)
#         return all_rets

# class TestCritic2(nn.Module):
#     """
#     Relational network, used as critic for all agents.
#     """
#
#     # TODO previously embedding_size = 16, now we have hidden_dim and 32
#     def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
#                  # obj_n: int,
#                  n_actions: int,
#                  input_dims: list,
#                  hidden_dim: object = 32,
#                  norm_in: object = True,
#                  net_code: object = "1g0f",
#                  mp_rounds: object = 1) -> object:
#         """
#         Inputs:
#             sa_sizes (list of (int, int)): Size of state and action spaces per
#                                           agent
#             hidden_dim (int): Number of hidden dimensions
#             norm_in (bool): Whether to apply BatchNorm to input
#         """
#         super(TestCritic2, self).__init__()
#         self.sa_sizes = sa_sizes
#         self.nagents = len(sa_sizes)
#         self.num_actions = n_actions[0]
#         self.critic_encoders = nn.ModuleList()
#         self.critics_head = nn.ModuleList()
#
#         # self.state_encoder = nn.ModuleList()
#         # iterate over agents
#         for _ in range(self.nagents):
#             critic = nn.Sequential()
#             critic.add_module('critic_fc1', nn.Linear(hidden_dim, # critic only takes in 1* hidden_dim now
#                                                       hidden_dim))
#             critic.add_module('critic_nl', nn.LeakyReLU())
#             critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
#             self.critics_head.append(critic) # one critic for each agent
#
#         embedder = nn.Linear(input_dims[1], hidden_dim) # TODO - hardcoding needed?
#
#         self.gcn = GCNConv(hidden_dim, hidden_dim)
#
#         self.embedder = embedder
#         self.shared_modules = []
#
#
#     def shared_parameters(self):
#         """
#         Parameters shared across agents and reward heads
#         """
#         return chain(*[m.parameters() for m in self.shared_modules])
#
#     def scale_shared_grads(self):
#         """
#         Scale gradients for parameters that are shared since they accumulate
#         gradients from the critic loss function multiple times
#         """
#         for p in self.shared_parameters():
#             p.grad.data.mul_(1. / self.nagents)
#
#     def forward(self,
#                 inps,
#                 unary_tensor,
#                 agents=None,
#                 return_q=True,
#                 return_all_q=False,
#                 regularize=False,
#                 logger=None,
#                 niter=0):
#         """
#         Inputs:
#             inps (list of PyTorch Matrices): Inputs to each agents' encoder
#                                              (batch of obs + ac)
#             agents (int): indices of agents to return Q for
#             return_q (bool): return Q-value
#             return_all_q (bool): return Q-value for all actions
#             regularize (bool): returns values to add to loss function for
#                                regularization
#             return_attend (bool): return attention weights per agent
#             logger (TensorboardX SummaryWriter): If passed in, important values
#                                                  are logged
#         """
#         states = [s for s, a in inps]
#         actions = [a for s, a in inps]
#
#         device = next(self.parameters()).device
#         if agents is None:
#             agents = range(self.nagents)
#         self.max_reduce = True
#
#         nb_objects = 16
#         batch_size = 64
#         slices = [nb_objects for _ in range(batch_size)]
#         #chunks = torch.split(embedds, slices, dim=0)  # splits it in slices/entities
#
#         adj_t = torch.ones([nb_objects, nb_objects]) - torch.diag(torch.ones(nb_objects))
#         edge_index = adj_t.nonzero().t().contiguous()
#
#         graph_list = []
#         feat_list =[]
#         for i in range(batch_size):
#             simple_array = np.array(states[0][i], dtype=np.int32)
#             encoded_array = np.zeros((simple_array.size,3), dtype=int)
#             encoded_array[np.arange(simple_array.size), simple_array] = 1
#             feat_tensor = torch.Tensor(encoded_array)
#             assert torch.equal(feat_tensor, unary_tensor[i])
#             feat_list.append(feat_tensor)
#             feat_tensor = torch.arange(nb_objects).view(-1, 1)
#             single = GeometricData(x=feat_tensor, edge_index=edge_index)
#             graph_list.append(single)
#         gd = Batch.from_data_list(graph_list)
#         x = np.vstack(feat_list)
#         x = torch.Tensor(x)
#         x = self.embedder(unary_tensor.flatten(0,1))
#         embedds = self.gcn(x, gd.edge_index)
#         chunks = torch.split(embedds, slices, dim=0) # splits it in slices/entities
#         chunks = [p.unsqueeze(0) for p in chunks] # just adds back another dimension in the beginning
#         x = torch.cat(chunks, dim=0)
#         if self.max_reduce:
#             x, _ = torch.max(x, dim=1)
#         else:
#             x = torch.flatten(x, start_dim=1, end_dim=2)
#
#         agents = range(1)
#         all_rets = []
#         for i, a_i in enumerate(agents):
#             # extract state encoding for each agent that we're returning Q for
#             agent_rets = []
#             all_q = self.critics_head[a_i](x)
#             int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
#             q = all_q.gather(1, int_acs)
#             if return_q:
#                 agent_rets.append(q)
#             if return_all_q:
#                 agent_rets.append(all_q)
#             # if logger is not None:
#             #    pass
#             # logger.add_scalars('agent%i/attention' % a_i,
#             #                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
#             #                        in enumerate(head_entropies)),
#             #                   niter)
#             if len(agent_rets) == 1:
#                 all_rets.append(agent_rets[0])
#             else:
#                 all_rets.append(agent_rets)
#         #        if len(all_rets) == 1:
#         #            return all_rets[0]
#         #        else:
#         #            return all_rets
#         return all_rets


class RelationalCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """
    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                  #obj_n: int,
                 spatial_tensors,
                 batch_size: int,
                 n_actions: int,
                 input_dims: list,
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
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.num_actions = n_actions[0]
        self.critics_head = nn.ModuleList()

        self.batch_size = batch_size
        self.max_reduce = True # TODO hardcoded
        self.spatial_tensors = np.array(spatial_tensors)
        self.binary_batch = torch.tensor([self.spatial_tensors for _ in range(self.batch_size)])
        self.gd, self.slices = batch_to_gd(self.binary_batch)  # makes adjs geometric data usable for torch geometric
        self.nb_edge_types = len(spatial_tensors)


        self.embedder = nn.Linear(input_dims[0], hidden_dim) # TODO - hardcoding needed?
        self.gnn_layers = RGCNConv(hidden_dim, hidden_dim, self.nb_edge_types)

        # iterate over agents
        for _ in range(self.nagents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim + self.num_actions * (self.nagents-1),
                                                      hidden_dim))
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
            p.grad.data.mul_(1. / self.nagents)

    def forward(self,
                unary_tensors,
                # binary_tensor,
                actions,
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
        device = next(self.parameters()).device
        if agents is None:
            agents = range(self.nagents)
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
            embedds = torch.flatten(unary_tensors[a_i], 0, 1).float()
            #embedds = self.embedding_linear(embedds)
            embedds = self.embedder(embedds) # seems right so far


            # RGCN module
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
            other_actions = actions[-a_i]
            critic_in = torch.cat((x, other_actions), dim=1)
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



def batch_to_gd(batch: torch.Tensor):
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
    return geometric_batch, slices
