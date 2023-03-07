from Aurora.agent.geometric.util import batch_to_gd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import List
from torch_geometric.nn import RGCNConv, GCNConv
from torch_geometric.data import Data as GeometricData, Batch

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        assert (hidden_dim % attend_heads) == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = adim
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * hidden_dim,
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

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

    def forward(self, inps, agents=None, return_q=True, return_all_q=False,
                regularize=False, return_attend=False, logger=None, niter=0):
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
            agents = range(len(self.critic_encoders))
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]

        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]

        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]

        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2) # x_i
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(1).sum(1) # -((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            # TODO change here to cater for 2 player games! squeeze() --> squeeze(1) because otherwise (1024, 1, 1) is squeezed to hard
            agent_rets = []
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[a_i](critic_in)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = (attend_mag_reg,)
                agent_rets.append(regs)
            if return_attend:
                agent_rets.append(np.array(all_attend_probs[i]))
            if logger is not None:
                logger.add_scalars('agent%i/attention' % a_i,
                                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
                                        in enumerate(head_entropies)),
                                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets


class TestCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """

    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                 # obj_n: int,
                 n_actions: int,
                 input_dims: list,
                 hidden_dim: object = 32,
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
        super(TestCritic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.num_actions = n_actions[0]
        self.critic_encoders = nn.ModuleList()
        self.critics_head = nn.ModuleList()

        # self.state_encoder = nn.ModuleList()
        # iterate over agents
        for _ in range(self.nagents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim,  # critic only takes in 1* hidden_dim now
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic)  # one critic for each agent

        self.embedder = nn.Linear(16, hidden_dim)  # TODO test

        self.shared_modules = [self.embedder]
        # self.shared_modules = [self.key_extractors, self.selector_extractors,
        #                       self.value_extractors, self.critic_encoders]

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
                inps,
                agents=None,
                return_q=True,
                return_all_q=False,
                regularize=False,
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
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        # extract state-action encoding for each agent
        s_encodings = self.embedder(states[0]) # TODO hardcoded

        agents = range(1) # TODO hardcoded
        all_rets = []
        for i, a_i in enumerate(agents):
            # extract state encoding for each agent that we're returning Q for
            agent_rets = []
            all_q = self.critics_head[a_i](s_encodings)
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

class TestCritic2(nn.Module):
    """
    Relational network, used as critic for all agents.
    """

    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                 # obj_n: int,
                 n_actions: int,
                 input_dims: list,
                 hidden_dim: object = 32,
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
        super(TestCritic2, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.num_actions = n_actions[0]
        self.critic_encoders = nn.ModuleList()
        self.critics_head = nn.ModuleList()

        # self.state_encoder = nn.ModuleList()
        # iterate over agents
        for _ in range(self.nagents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim, # critic only takes in 1* hidden_dim now
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent

        embedder = nn.Linear(input_dims[1], hidden_dim) # TODO - hardcoding needed?

        self.gcn = GCNConv(hidden_dim, hidden_dim)

        self.embedder = embedder
        self.shared_modules = []


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
                inps,
                unary_tensor,
                agents=None,
                return_q=True,
                return_all_q=False,
                regularize=False,
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
        states = [s for s, a in inps]
        actions = [a for s, a in inps]

        device = next(self.parameters()).device
        if agents is None:
            agents = range(self.nagents)
        self.max_reduce = True

        nb_objects = 16
        batch_size = 64
        slices = [nb_objects for _ in range(batch_size)]
        #chunks = torch.split(embedds, slices, dim=0)  # splits it in slices/entities

        adj_t = torch.ones([nb_objects, nb_objects]) - torch.diag(torch.ones(nb_objects))
        edge_index = adj_t.nonzero().t().contiguous()

        graph_list = []
        feat_list =[]
        for i in range(batch_size):
            simple_array = np.array(states[0][i], dtype=np.int32)
            encoded_array = np.zeros((simple_array.size,3), dtype=int)
            encoded_array[np.arange(simple_array.size), simple_array] = 1
            feat_tensor = torch.Tensor(encoded_array)
            assert torch.equal(feat_tensor, unary_tensor[i])
            feat_list.append(feat_tensor)
            feat_tensor = torch.arange(nb_objects).view(-1, 1)
            single = GeometricData(x=feat_tensor, edge_index=edge_index)
            graph_list.append(single)
        gd = Batch.from_data_list(graph_list)
        x = np.vstack(feat_list)
        x = torch.Tensor(x)
        x = self.embedder(unary_tensor.flatten(0,1))
        embedds = self.gcn(x, gd.edge_index)
        chunks = torch.split(embedds, slices, dim=0) # splits it in slices/entities
        chunks = [p.unsqueeze(0) for p in chunks] # just adds back another dimension in the beginning
        x = torch.cat(chunks, dim=0)
        if self.max_reduce:
            x, _ = torch.max(x, dim=1)
        else:
            x = torch.flatten(x, start_dim=1, end_dim=2)

        agents = range(1)
        all_rets = []
        for i, a_i in enumerate(agents):
            # extract state encoding for each agent that we're returning Q for
            agent_rets = []
            all_q = self.critics_head[a_i](x)
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)
            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            # if logger is not None:
            #    pass
            # logger.add_scalars('agent%i/attention' % a_i,
            #                   dict(('head%i_entropy' % h_i, ent) for h_i, ent
            #                        in enumerate(head_entropies)),
            #                   niter)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)
        #        if len(all_rets) == 1:
        #            return all_rets[0]
        #        else:
        #            return all_rets
        return all_rets


class RelationalCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """
    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                 #obj_n: int,
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
        self.critic_encoders = nn.ModuleList()
        self.critics_head = nn.ModuleList()

        # self.state_encoder = nn.ModuleList()
        # iterate over agents
        for _ in range(self.nagents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim, # critic only takes in 1* hidden_dim now
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent


        nb_edge_types = input_dims[2]

        # default is nb_layers = gnn_layers = 2 and nb_dense_layers =0
        nb_layers, nb_dense_layers, _ = parse_code(net_code)
        self.max_reduce = True # TODO hardcoded

        embedder = nn.Linear(input_dims[1], hidden_dim) # TODO - hardcoding needed?

        self.gnn_layers = RGCNConv(hidden_dim, hidden_dim, nb_edge_types)
        self.embedder = embedder # TODO shared or individual?

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
                unary_tensor,
                binary_tensor,
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
        state = None # TODO change
        inputs = [[],
                  torch.flatten(unary_tensor, 0, 1).float(), #flattens obs["unary_tensor"] only in 0th and 1st dim
                  #torch.flatten(binary_tensor, 0, 1).permute(0,3,1,2).float()]
                  binary_tensor.permute(0, 3, 1, 2).float()
                  ]
        for i in [1,2]:
            inputs[i] = inputs[i].to(device=device)
        adj_matrices = inputs[2]
        gd, slices = batch_to_gd(adj_matrices) # makes adjs geometric data usable for torch geometric

        # feature embedding
        #embedds = torch.flatten(inputs[1], 0, 1)
        embedds = inputs[1]
        #embedds = self.embedding_linear(embedds)
        embedds = self.embedder(embedds) # seems right so far


        # RGCN module
        embedds = self.gnn_layers(embedds, gd.edge_index, gd.edge_attr).relu()
        #embedds = torch.relu(embedds)

        chunks = torch.split(embedds, slices, dim=0) # splits it in slices/entities
        chunks = [p.unsqueeze(0) for p in chunks] # just adds back another dimension in the beginning
        x = torch.cat(chunks, dim=0)
        if self.max_reduce:
        # max-pooling layer
            x, _ = torch.max(x, dim=1) # TODO check what dimension comes out of here
        else:
            # I think this would be for the CNN which is flattened
            x = torch.flatten(x, start_dim=1, end_dim=2) # TODO what does that do?


        all_rets = []
        for i, a_i in enumerate(agents):
        # extract state encoding for each agent that we're returning Q for
            agent_rets = []
            all_q = self.critics_head[a_i](x)
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
class GCNCritic(nn.Module):
    """
    Relational network, used as critic for all agents.
    """
    # TODO previously embedding_size = 16, now we have hidden_dim and 32
    def __init__(self, sa_sizes: list,  # TODO at the end we should not need sa_sizes anymore?
                 #obj_n: int,
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
        super(GCNCritic, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.num_actions = n_actions[0]
        self.critic_encoders = nn.ModuleList()
        self.critics_head = nn.ModuleList()

        # self.state_encoder = nn.ModuleList()
        # iterate over agents
        for _ in range(self.nagents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim, # critic only takes in 1* hidden_dim now
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent

        #embedder = nn.Linear(input_dims[1], hidden_dim) # TODO - hardcoding needed?

        self.gcn = GCNConv(input_dims[1], hidden_dim)

        #self.embedder = embedder # TODO shared or individual?
        self.shared_modules = []

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
                unary_tensor,
                binary_tensor,
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
        self.max_reduce = True
        inputs = [[],
                  torch.flatten(unary_tensor, 0, 1).float(), #flattens obs["unary_tensor"] only in 0th and 1st dim
                  binary_tensor.permute(0, 3, 1, 2).float()
                  ]
        for i in [1,2]:
            inputs[i] = inputs[i].to(device=device)

        # feature embedding
        #embedds = self.embedder(inputs[1]) # seems right so far
        embedds = inputs[1]


        nb_relations = 1
        nb_objects = 16
        batch_size = 64

        adj_t = torch.ones([nb_objects, nb_objects]) - torch.diag(torch.ones(nb_objects))
        edge_index = adj_t.nonzero().t().contiguous()

        graph_list = []
        for i in range(batch_size):
            x = torch.arange(nb_objects).view(-1, 1)
            single = GeometricData(x=x, edge_index=edge_index)
            graph_list.append(single)
        gd = Batch.from_data_list(graph_list)

        embedds = self.gcn(embedds, gd.edge_index)
        slices = [nb_objects for _ in range(batch_size)]
        chunks = torch.split(embedds, slices, dim=0) # splits it in slices/entities
        chunks = [p.unsqueeze(0) for p in chunks] # just adds back another dimension in the beginning
        x = torch.cat(chunks, dim=0)
        x, _ = torch.max(x, dim=1) # TODO check what dimension comes out of here


        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            all_q = self.critics_head[a_i](x)
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



