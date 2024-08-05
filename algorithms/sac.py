import torch
import wandb
from torch.optim import Adam
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import RelationalCritic, AttentionCritic
from utils.base_critic import BaseCritic
import numpy as np
from gym.spaces import Dict

MSELoss = torch.nn.MSELoss()

class RelationalSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self,
                 agent_init_params,
                 spatial_tensors,
                 batch_size,
                 n_actions,
                 input_dims,
                 n_agents=2,
                 gamma=0.95,
                 tau=0.01,
                 pi_lr=0.01,
                 q_lr=0.01,
                 reward_scale=10.,
                 embed_size = 128,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128,
                 graph_layer='RGCN', 
                 device='cuda:0',
                 dense= True,
                 net_code = '1g1i1f',
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent, input size (observation shape) and output size (action shape)
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        # print('initalised sac with ', device)
        self.n_agents = n_agents
        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      num_in_pol=params['num_in_pol'],
                                      num_out_pol=params['num_out_pol'])
                         for params in agent_init_params] # are input and output dims for agent
        print('loading from', device)
        self.critic = RelationalCritic(n_agents=self.n_agents,
                                       spatial_tensors=spatial_tensors,
                                       batch_size = batch_size,
                                       n_actions=n_actions,
                                       input_dims=input_dims,
                                       hidden_dim=critic_hidden_dim,
                                       graph_layer = graph_layer,
                                       device = device,
                                       dense = dense,
                                       net_code = net_code,
                                       embed_size = embed_size,
                                       )
        self.target_critic = RelationalCritic(
                                        n_agents = self.n_agents,
                                        spatial_tensors=spatial_tensors,
                                        batch_size = batch_size,
                                        n_actions=n_actions,
                                        input_dims=input_dims,
                                        hidden_dim=critic_hidden_dim,
                                        graph_layer = graph_layer, 
                                        device=device,
                                        dense = dense,
                                        net_code = net_code,
                                        embed_size = embed_size,
                                        )
        hard_update(self.target_critic, self.critic) # hard update only at the beginning to initialise
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params #in: obs.shape out: action.shape
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.device = device
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics


    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        # observations = [observations['image']] * 2
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def target_step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        # observations = [observations['image']] * 2
        return [a.target_step(obs) for a, obs in zip(self.agents, observations)]

    def critic_embeds(self, obs, acs):
        unary = [o['unary_tensor'] for o in obs]
        binary = [[o['binary_tensor']] for o in obs]
        return self.critic(obs=obs, unary_tensors=unary, binary_tensors=binary, actions=acs)


    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, unary,binary, acs, rews, next_obs, next_unary, next_binary, dones = sample

        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)


        for n, p in self.critic.named_parameters():
            if 'gnn_layers' in n:
                # print(n)
                p.requires_grad = False

        for n, p in self.target_critic.named_parameters():
            if 'gnn_layers' in n:
                p.requires_grad = False

        critic_rets = self.critic(obs=obs, unary_tensors=unary, binary_tensors=binary, actions=acs)
        next_qs = self.target_critic(obs=next_obs, unary_tensors=next_unary, binary_tensors=next_binary, actions=next_acs)
        q_loss = 0
        for a_i, nq, log_pi, pq in zip(range(self.n_agents), next_qs,
                                               next_log_pis, critic_rets):
            target_q = (rews[a_i].view(-1, 1) +
                        self.gamma * nq *
                        (1 - dones[a_i].view(-1, 1)))
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())

        q_loss.backward()


        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.n_agents)

        wandb.log({'grad_norm': grad_norm, 'q_loss': q_loss})
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, unary, binary, acs, rews, next_obs, next_unary, next_binary,  dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.n_agents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)

            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs) # TODO reg unknown?

        critic_rets = self.critic(obs=obs, unary_tensors=unary, binary_tensors=binary, actions=samp_acs,
                                  logger=logger, return_all_q=True)

        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.n_agents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()




    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='cuda:0'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        fn = lambda x: x.to(device)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.target_policy.eval()
        fn = lambda x: x.to(device)
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename, episode = None):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()},
                     'episode': episode,
                     'device':self.device}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env,
                      spatial_tensors,
                      batch_size,
                      dense,
                      gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                      graph_layer = 'RGCN',
                      embed_size = 128,
                      pol_hidden_dim=64,
                      critic_hidden_dim=64,
                      net_code = '1g1i1f',
                      device='cuda:0',
                      **kwargs):
        """
        Instantiate instance of this class from multi-agent environment

        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        print(device, 'device for critic')
        agent_init_params = []
        a_size = []
        for acsp, obsp in zip(env.action_space,
                              env.observation_space):
            print(type(obsp))
            if isinstance(obsp, dict) or isinstance(obsp, Dict):
                agent_init_params.append({'num_in_pol': np.ones(shape=obsp['image'].shape).flatten().shape[0],
                                          'num_out_pol': acsp.n})
            else:
               agent_init_params.append({'num_in_pol': np.ones(shape=obsp.shape).shape[0],
                                         'num_out_pol': acsp.n})
            a_size.append(acsp.n)

        init_dict = {'gamma': gamma,
                     'tau': tau,
                     'pi_lr': pi_lr,
                     'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'agent_init_params': agent_init_params,
                     'n_agents': env.n_agents,
                     'spatial_tensors': spatial_tensors,
                     'batch_size': batch_size,
                     'n_actions': a_size,
                     'input_dims': [env.n_attr, env.n_rel_rules],
                     'device':device,
                     'graph_layer':graph_layer,
                     'dense': dense,
                     'net_code': net_code,
                     'embed_size': embed_size,
                     }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False, device='cuda:0'):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        # if torch.cuda.is_available():
        #     save_dict = torch.load(filename, map_location="cuda")
        # else:
        save_dict = torch.load(filename, map_location="cpu")
        try:
            episode = save_dict['episode']
        except Exception as e:
            print(e)
            episode = 0
        save_dict['init_dict']['device'] = device
        print(save_dict['init_dict'])
        # save_dict['init_dict']['n_agents'] = 7
        # save_dict['init_dict']['agent_init_params'] = [save_dict['init_dict']['agent_init_params'][0] for _ in range(save_dict['init_dict']['n_agents'])]
        # save_dict['agent_params'] = save_dict['agent_params'] * save_dict['init_dict']['n_agents']
        # save_dict['agent_params'].pop(0) # if less agents
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params, device=device)
        instance.pol_dev = device
        instance.trgt_pol_dev = device
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic = instance.critic.to(device)
            instance.target_critic = instance.target_critic.to(device)
            instance.critic_optimizer = Adam(instance.critic.parameters(), lr=0.01, weight_decay=1e-3)
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            instance.critic_dev = device
            instance.trgt_critic_dev = device

        return instance, episode

class AttentionSAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.,
                 pol_hidden_dim=128,
                 critic_hidden_dim=128, attend_heads=4,
                 hard=True,
                 base =True,
                 **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.base = base
        self.n_agents = len(sa_size)
        self.nobjects = None
        self.agents = [AttentionAgent(lr=pi_lr,
                                      hidden_dim=pol_hidden_dim,
                                      **params)
                         for params in agent_init_params]
        if self.base:
            self.critic = BaseCritic(sa_size, hidden_dim=critic_hidden_dim)
            self.target_critic = BaseCritic(sa_size, hidden_dim=critic_hidden_dim)
        else:
            self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                          attend_heads=attend_heads, hard=hard)
            self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim,
                                                 attend_heads=attend_heads, hard=hard)

        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr,
                                     weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def target_step(self, observations):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.target_step(obs) for a, obs in zip(self.agents, observations)]

    def critic_embeds(self, obs, acs):
        critic_in = list(zip(obs, acs))
        critic_rets = self.critic(critic_in, regularize=True)
        return critic_rets


    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic for all agents
        """
        obs, acs, rews, next_obs, dones = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in) # gives us the single q-value given observation and action of agent
        critic_rets = self.critic(critic_in, regularize=True)
        q_loss = 0
        if self.base:
            for a_i, nq, log_pi, pq in zip(range(self.n_agents), next_qs,
                                                   next_log_pis, critic_rets):
                target_q = (rews[a_i].view(-1, 1) +
                            self.gamma * nq *
                            (1 - dones[a_i].view(-1, 1)))
                if soft:
                    target_q -= log_pi / self.reward_scale # reward scale is the alpha!
                q_loss += MSELoss(pq, target_q.detach())
            q_loss.backward()
        else:
            for a_i, nq, log_pi, (pq, regs) in zip(range(self.n_agents), next_qs,
                                                   next_log_pis, critic_rets):
                target_q = (rews[a_i].view(-1, 1) +
                            self.gamma * nq *
                            (1 - dones[a_i].view(-1, 1)))
                if soft:
                    target_q -= log_pi / self.reward_scale # reward scale is the alpha!
                q_loss += MSELoss(pq, target_q.detach())
                # print(q_loss.grad_fb)
                for reg in regs:
                    q_loss += reg  # regularizing attention
            q_loss.backward()
            self.critic.scale_shared_grads()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.n_agents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()


    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.n_agents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(
                ob, return_all_probs=True, return_log_pi=True,
                regularize=True, return_entropy=True)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs, (q, all_q) in zip(range(self.n_agents), all_probs,
                                                            all_log_pis, all_pol_regs,
                                                            critic_rets):
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()



    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='cuda:0'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        fn = lambda x: x.to(device)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
            a.target_policy.eval()
        fn = lambda x: x.to(device)
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename, episode = None):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'critic_params': {'critic': self.critic.state_dict(),
                                       'target_critic': self.target_critic.state_dict(),
                                       'critic_optimizer': self.critic_optimizer.state_dict()},
                     'episode': episode}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01,
                      pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.,
                     pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4,
                     hard = True, 
                      **kwargs):

        """
        Instantiate instance of this class from multi-agent environment
        env: Multi-agent Gym environment
        gamma: discount factor
        tau: rate of update for target networks
        lr: learning rate for networks
        hidden_dim: number of hidden dimensions for networks
        """
        agent_init_params = []
        sa_size = []

        if env.grid_observation:
            for acsp, obsp in zip(env.action_space,
                                  env.observation_space):
                # s_shape = np.prod(obsp['image'].shape)
                s_shape = np.prod(obsp['image'].shape)
                agent_init_params.append({'num_in_pol': s_shape,
                                          'num_out_pol': acsp.n})
                sa_size.append((s_shape, acsp.n))
        else:
            for acsp, obsp in zip(env.action_space,
                                  env.observation_space):
                    agent_init_params.append({'num_in_pol': obsp.shape[0],
                                              'num_out_pol': acsp.n})
                    sa_size.append((obsp.shape[0], acsp.n))


        init_dict = {'gamma': gamma, 'tau': tau,
                     'pi_lr': pi_lr, 'q_lr': q_lr,
                     'reward_scale': reward_scale,
                     'pol_hidden_dim': pol_hidden_dim,
                     'critic_hidden_dim': critic_hidden_dim,
                     'attend_heads': attend_heads,
                     'hard': hard, 
                     'agent_init_params': agent_init_params,
                     'sa_size': sa_size}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False, device='cuda:0'):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        try:
            episode = save_dict['episode']
        except Exception as e:
            print(e)
            episode = 0
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params, device=device)
        instance.pol_dev = device
        instance.trgt_pol_dev = device
        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.critic = instance.critic.to(device)
            instance.target_critic = instance.target_critic.to(device)


            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            instance.critic_dev = device
            instance.trgt_critic_dev = device

        return instance, episode
