import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from torch_geometric.nn import RGCNConv, pool, RGATConv, GATv2Conv, GATConv, Sequential
from torch_geometric.data import Data as GeometricData, Batch

class BaseCritic(nn.Module):
    """
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """
        super(BaseCritic, self).__init__()
        self.n_agents = len(sa_sizes)
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in sa_sizes:
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim + adim * (self.n_agents-1),
                                                      hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, adim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

    def forward(self, inps, agents=None, return_q=True, return_all_q=False, regularize=False):
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
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            other_actions = actions.copy()
            other_actions.pop(a_i)
            critic_in = torch.cat((s_encodings[i], *other_actions), dim=1)
            all_q = self.critics[a_i](critic_in)
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
        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets

