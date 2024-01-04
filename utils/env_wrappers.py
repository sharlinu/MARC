"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper
import gym


class FlatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.env = env
        self.observation_space = None
        dim = self.n_objects * 2 * 2 + self.n_agents * 4
        self.single_space = gym.spaces.Box(np.ones(dim)*-1, np.ones(dim)*100, dtype=np.float32)
        self.observation_space = [self.single_space for _ in range(self.n_agents)]

    def observation(self, obs):
        """
        Flattens the observation dictionary into a 1D numpy array

        Args:
        obs (dict): The observation dictionary.

        Returns:
        list: List of flattened observation array as np.ndarray
        """
        flattened_obs_all = []
        num_agents = len(obs)

        # Information for each agent
        for i in range(num_agents):
            flattened_obs = []
            agent_key = f'agent_{i}'
            agent_obs = obs[agent_key]

            # Self information
            flattened_obs.extend(agent_obs['self']['position'])
            flattened_obs.append(int(agent_obs['self']['picker']))
            flattened_obs.append(agent_obs['self']['carrying_object']
                                 if agent_obs['self']['carrying_object'] is not None else -1)

            # Other agents' information
            for other_agent in agent_obs['agents']:
                flattened_obs.extend(other_agent['position'])
                flattened_obs.append(int(other_agent['picker']))
                flattened_obs.append(
                    other_agent['carrying_object'] if other_agent['carrying_object'] is not None else -1)

            # Objects' information
            for obj in obs[agent_key]['objects']:
                flattened_obs.extend(obj['position'])

            # Goals' information
            for goal in obs[agent_key]['goals']:
                flattened_obs.extend(goal)
            flattened_obs = np.array(flattened_obs)
            flattened_obs_all.append(flattened_obs)

        return flattened_obs_all

class GridObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = None
        single_observation_space = {}
        single_observation_space['image'] = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(*env.grid_size, 6),
            dtype=np.float32)
        self.observation_space = [single_observation_space for _ in range(self.n_agents)]
    def observation(self, obs):
        '''
        Converts the observation dictionary into a 3D grid representation.

        The grid is represented as a 3D NumPy array with dimensions (grid_width, grid_length, 3),
        where the last dimension corresponds to different channels for agents, objects, and goals.
        Each cell in the grid can be either 0 or 1, indicating the absence or presence of an entity.

        Args:
        obs (dict): The observation dictionary containing information about agents, objects, and goals.
        grid_size (tuple): A tuple representing the size of the grid as (grid_width, grid_length).

        Returns:
        np.ndarray: A 3D NumPy array representing the grid.
        '''
        obs_all = []

        for _, agent_data in obs.items():
            single_obs = {}

            grid = np.zeros((*self.grid_size, 6))
            x, y = agent_data['self']['position']
            grid[x, y, 3] = 1  # ID layer
            grid[x, y, 0] = 1
            grid[x, y, 4] = 1 if agent_data['self']['carrying_object'] is not None else -1
            grid[x, y, 5] = agent_data['self']['picker']

            for other_agent in agent_data['agents']:
                x, y = other_agent['position']
                grid[x, y, 0] = 1
                grid[x, y, 4] = 1 if other_agent['carrying_object'] is not None else -1
                grid[x, y, 5] = other_agent['picker']

            for obj in agent_data['objects']:
                x, y = obj['position']
                grid[x, y, 1] = 1

            for goal in agent_data['goals']:
                x, y = goal
                grid[x, y, 2] = 1

            single_obs['image'] = grid
            obs_all.append(single_obs)

        return (obs_all)


class PartialGridObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = None
        single_observation_space = {}
        a, b, c = env.observation_space[0].shape
        obs_shape = (b, c, a)
        single_observation_space['image'] = gym.spaces.Box(
            low=np.zeros(obs_shape),
            high=np.ones(obs_shape),
            shape=obs_shape,
            dtype=np.float32)
        self.observation_space = [single_observation_space for _ in range(self.n_agents)]

    def observation(self, observation):
        obs_all = []
        for obs in observation:
            single_obs = {}
            obs = np.moveaxis(obs, -1, 0)
            single_obs['image'] = obs
            obs_all.append(single_obs)
        return (obs_all)



def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError

class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        """
        Steps through all environment and resets individual ones that have terminated
        Returns
        Standard gym quadruplet for step function
        -------

        """
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return



