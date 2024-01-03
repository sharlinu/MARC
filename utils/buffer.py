import numpy as np
from torch import Tensor
from torch.autograd import Variable
from collections import deque

class ReplayBufferMAAC(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):\
            # first index agent, 2nd index replay_length
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i]) #np.vstack not needed with just one environment
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, device='cpu', norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        # all this does is turn sample into a tensor
        
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std()) if self.rew_buffs[i][:self.filled_i].mean() != 0
                        else cast(self.rew_buffs[i][inds])
                        for i in range(self.num_agents)]
            # print('ret_rews', ret_rews)
            # print('mean, std 0', self.rew_buffs[0][:self.filled_i].mean(), self.rew_buffs[0][:self.filled_i].std())
            # print('mean, std 1', self.rew_buffs[1][:self.filled_i].mean(), self.rew_buffs[1][:self.filled_i].std())
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]


class ReplayBufferMARC(object):
    """
    Replay Buffer for multi-agent RL with stored graph data
    """

    def __init__(self,
                 max_steps,
                 num_agents,
                 obs_dims,
                 # unary_dims,
                 # binary_dims,
                 ac_dims,
                 dense= True):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.obs_unary_buffs = [deque([], maxlen=max_steps) for _ in range(self.num_agents)]
        self.obs_binary_buffs = [deque([], maxlen=max_steps) for _ in range(self.num_agents)]
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        # self.next_obs_unary_buffs = []
        self.next_obs_binary_buffs = [deque([], maxlen=max_steps) for _ in range(self.num_agents)]
        self.next_obs_unary_buffs = [deque([], maxlen=max_steps) for _ in range(self.num_agents)]
        self.done_buffs = []
        self.dense = dense
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            # self.obs_unary_buffs.append(np.zeros((max_steps,) + unary_dim, dtype=np.float32))

            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))

            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))

            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))


            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (overwrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self,
             observation: tuple,
             actions: list,
             rewards,
             next_observation: tuple,
             dones):
        nentries = 1    # usually for multiple parallel environments, but now just one anyways

        if self.curr_i + nentries > self.max_steps:
            print('rollover')
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                # self.obs_unary_buffs[agent_i] = np.roll(self.obs_unary_buffs[agent_i],
                #                                   rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)

                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                # self.next_obs_unary_buffs[agent_i] = np.roll(
                #     self.next_obs_unary_buffs[agent_i], rollover, axis=0)

                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observation[agent_i]['image'].flatten()
            # self.obs_unary_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observation[agent_i]['unary_tensor']

            if self.dense:
                # print('obs g', observation[agent_i]['binary_tensor'])
                # print('next_obs g', next_observation[agent_i]['binary_tensor'])
                try:
                    self.obs_binary_buffs[agent_i].insert(self.curr_i, observation[agent_i]['binary_tensor'])
                    self.next_obs_binary_buffs[agent_i].insert(self.curr_i, next_observation[agent_i]['binary_tensor'])
                    # self.obs_unary_buffs[agent_i].insert(self.curr_i, observation[agent_i]['unary_tensor'])
                    # self.next_obs_unary_buffs[agent_i].insert(self.curr_i, next_observation[agent_i]['unary_tensor'])
                except IndexError:
                    self.obs_binary_buffs[agent_i][self.curr_i] = observation[agent_i]['binary_tensor']
                    self.next_obs_binary_buffs[agent_i][self.curr_i] = next_observation[agent_i]['binary_tensor']
                # print('binary buffer length:', len(self.obs_binary_buffs[0]), len(self.next_obs_binary_buffs[0]))
                # print('general buffer length', self.curr_i, self.filled_i)
            else:
                try:
                    self.next_obs_unary_buffs[agent_i].insert(self.curr_i, next_observation[agent_i]['unary_tensor'].numpy())
                    self.obs_unary_buffs[agent_i].insert(self.curr_i, observation[agent_i]['unary_tensor'].numpy())
                except IndexError:
                    self.obs_unary_buffs[agent_i][self.curr_i] = observation[agent_i]['unary_tensor'].numpy()
                    self.next_obs_unary_buffs[agent_i][self.curr_i] = next_observation[agent_i]['unary_tensor'].numpy()

            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]

            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observation[agent_i]['image'].flatten()


            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, device='cpu', norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std()) if self.rew_buffs[i][:self.filled_i].mean() != 0
                        else cast(self.rew_buffs[i][inds])
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        if self.dense:
            out = ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [None for i in range(self.num_agents)], # seperate unary tensors
                    # [cast([self.obs_unary_buffs[i][ind] for ind in inds]) for i in range(self.num_agents)],
                    [[self.obs_binary_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [None for i in range(self.num_agents)],
                    # [cast([self.next_obs_unary_buffs[i][ind] for ind in inds]) for i in range(self.num_agents)],
                    [[self.next_obs_binary_buffs[i][ind] for ind in inds] for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])
        else:
            out =  ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast([self.obs_unary_buffs[i][ind] for ind in inds]) for i in range(self.num_agents)],
                    # [[cast(self.obs_unary_buffs[i][ind]) for ind in inds] for i in range(self.num_agents)],
                    [None for i in range(self.num_agents)], # graphs
                    [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                    ret_rews,
                    [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                    [cast([self.next_obs_unary_buffs[i][ind] for ind in inds]) for i in range(self.num_agents)],
                    # [[cast(self.next_obs_unary_buffs[i][ind]) for ind in inds] for i in range(self.num_agents)],
                    [None for i in range(self.num_agents)],
                    [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])
        return out

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]


# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward'))

