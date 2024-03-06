import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from torch_geometric.nn import RGCNConv, pool, RGATConv, GATv2Conv, GATConv, Sequential
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
                 net_code: str = "1g1i1f",
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
        self.n_agents = n_agents
        self.num_actions = n_actions[0]
        self.critics_head = nn.ModuleList()
        self.device = device
        self.batch_size = batch_size
        self.max_reduce = True # TODO hardcoded
        self.embed_size = 16
        self.dense = dense

        self.nb_graph_layers, self.nb_iterations, self.nb_dense_layers = parse_code(net_code)
        self.graph_layer = graph_layer
        self.spatial_tensors = np.array(spatial_tensors)
        self.binary_batch = torch.tensor([self.spatial_tensors for _ in range(self.batch_size)])
        self.gd, self.slices = batch_to_gd(self.binary_batch, self.device)  # makes adjs geometric data usable for torch geometric
        self.nb_edge_types = len(spatial_tensors)
        self.embedder = nn.Linear(input_dims[0], self.embed_size)
        if self.graph_layer == 'RGCN':
            input_args = 'x, edge_index, edge_attr'
            gnn = (RGCNConv(self.embed_size, self.embed_size, self.nb_edge_types), 'x, edge_index, edge_attr -> x')
        elif self.graph_layer == 'RGAT':
            input_args = 'x, edge_index, edge_attr'
            attend_heads = 1
            assert (self.embed_size % attend_heads) == 0
            attend_dim = self.embed_size // attend_heads
            gnn = (RGATConv(in_channels=self.embed_size,
                                       out_channels=attend_dim,
                                       num_relations = self.nb_edge_types,
                                       heads = attend_heads,
                                       ),
                   'x, edge_index, edge_attr -> x')
            print('Using RGAT as graph layer')
        elif self.graph_layer == 'GAT':
            input_args = 'x, edge_index'
            attend_heads = 4
            assert (self.embed_size % attend_heads) == 0
            attend_dim = self.embed_size // attend_heads

            gnn = (GATConv(in_channels = input_dims[0],
                                      out_channels=attend_dim,
                                      heads=attend_heads,
                                      v2 = True
                                      ), 'x, edge_index -> x')
        elif self.graph_layer == 'GATv2':
            input_args = 'x, edge_index'
            attend_heads = 1
            assert (self.embed_size % attend_heads) == 0
            attend_dim = self.embed_size // attend_heads
            gnn = (GATv2Conv(in_channels=input_dims[0],
                                      out_channels=attend_dim,
                                      heads=attend_heads,
                                      ), 'x, edge_index -> x')
            print('Using GATv2 layer')
        else:
            print('not a valid graph layer')
        print(f'Using {self.graph_layer} as graph layer')
        gnn_list = []
        print(f'---- Adding {self.nb_graph_layers} layers and {self.nb_iterations} iterations ----')
        for i in range(self.nb_graph_layers):
            gnn_list.append(gnn)
            gnn_list.append(torch.nn.ReLU(inplace=True))
        self.gnn_layers = Sequential(input_args, gnn_list)
        # iterate over agents
        self.pooling = Pool()
        for _ in range(self.n_agents):
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(self.embed_size + self.num_actions * (self.n_agents-1),
                                                      hidden_dim)) # takes in 128+6*2 , out 128
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, self.num_actions))
            self.critics_head.append(critic) # one critic for each agent

        self.shared_modules = [self.embedder, self.gnn_layers]
        # self.shared_modules = [self.gnn_layers]

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
                return_all_q=False,
                logger = None,
                niter = 1):
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
                batch = gd.batch
                if self.graph_layer in ['RGCN', 'RGAT']:
                    embedds = self.embedder(gd.x)
                    for _ in range(self.nb_iterations):
                        embedds = self.gnn_layers(embedds, gd.edge_index, gd.edge_attr)
                else:
                    embedds = self.gnn_layers(gd.x, gd.edge_index)

            else:
                embedds = torch.flatten(unary_tensors[a_i], 0, 1).float().to(device=self.device)
                embedds = self.embedder(embedds)
                for _ in range(self.nb_iterations):
                    embedds = self.gnn_layers(embedds, self.gd.edge_index, self.gd.edge_attr)
                batch = self.gd.batch
            x = pool.global_max_pool(embedds, batch)

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

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per agent
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
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(1).sum(1) # -((probs + 1e-8).log() * probs).squeeze().sum(1)
                               .mean()) for probs in all_attend_probs[i]]
            # 2 agents: squeeze(1) gives (1024,1), whereas squeeze() gives (1024,)
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

def parse_code(net_code: str):
    """
    :param net_code: format <a>g<b>i<c>f
    """
    assert net_code[1]=="g"
    assert net_code[-1]=="f"
    nb_gnn_layers = int(net_code[0])
    nb_iterations = int(net_code[2])
    nb_dense_layers = int(net_code[-2])
    return nb_gnn_layers, nb_iterations, nb_dense_layers

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
