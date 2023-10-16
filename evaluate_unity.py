import argparse
import torch
import time
import imageio
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import RelationalSAC, AttentionSAC
import os
import json
import sys
from utils.misc import Agent

from Gridworld_Scripts.Connection import Connection as Conn
from Gridworld_Scripts.UnityGridEnv import UnityGridEnv as Env 

from lbforaging.foraging import ForagingEnv
import numpy as np
from enum import Enum
import yaml
from lbforaging.foraging import ForagingEnv
from utils.env_wrappers import DummyVecEnv

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

def make_parallel_MAAC_env(env, config):
    from utils.env_wrappers import DummyVecEnv
    def get_env_fn():
        def init_env():
            env.agents = [Agent() for _ in range(config.player)]
            # env.grid_observation = args.grid_observation
            # env.seed(args.random_seed + rank * 1000)
            np.random.seed(config.random_seed)
            return env
        return init_env
    # if config.n_rollout_threads == 1:
    return DummyVecEnv([get_env_fn()])

def run(config):
    model_path = config.model_path

    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    gif_path = '{}/{}'.format(eval_path, 'gifs')
    os.makedirs(gif_path, exist_ok=True)

    if config.alg=='MARC':
        model, _ = RelationalSAC.init_from_save(model_path)
    else:
        model, _ = AttentionSAC.init_from_save(model_path)
    print(config.benchmark)
    conn = Conn([''])
    env = Env(0, conn)
    # if config.alg == 'MAAC':
    #     env = make_parallel_MAAC_env(env, config)
    model.prep_rollouts(device='cpu')

    ifi = 1 / config.fps  # inter-frame interval
    collect_data = {}
    l_ep_rew = []
    for ep_i in range(config.test_n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.test_n_episodes))

        frames = []
        collect_item = {
            'ep': ep_i,
            'final_reward': 0,
            'l_infos': [],
            'l_rewards': []
        }
        l_rewards = []
        ep_rew = 0

        # from utils.rel_wrapper2 import AbsoluteVKBWrapper
        # env = AbsoluteVKBWrapper(env, config.dense, background_id=config.background_id)

        obs = env.reset()
        #if render:
        #    env.render()

        #for t_i in range(config.test_episode_length):
        for t_i in range(15):    
            calc_start = time.time()

            # rearrange observations to be per agent, and convert to torch Variable
            if config.grid_observation:
                obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(model.n_agents)]
            # get actions as torch Variables
            torch_actions = model.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
            # print('actions', actions)
            obs, rewards, dones, infos = env.step(actions)
            rewards, dones = np.array(rewards), np.array(dones)

            if render:
            #    env.render()
                time.sleep(1)
            collect_item['l_infos'].append(infos)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if render and (elapsed < ifi):
                time.sleep(ifi - elapsed)
            ep_rew += sum(rewards)

            if dones.any():
                collect_item['finished'] = 1
                break


        l_ep_rew.append(ep_rew)
        print("Reward: {}".format(ep_rew))

        collect_data[ep_i] = collect_item

        with open('{}/collected_data.json'.format(eval_path), 'w') as outfile:
            json.dump(collect_data, outfile,indent=4)

    print("Average reward: {}".format(sum(l_ep_rew)/config.test_n_episodes))
    env.close()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model_path")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--test_n_episodes", default=10, type=int)
    parser.add_argument("--test_episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--no_render", default=True, action="store_false",
                        help="render")
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    config = parser.parse_args()
    render = True
    args = vars(config)
    eval_path = Path(config.model_path)
    dir_exp = Path(*eval_path.parts[:-2])
    with open(f"{dir_exp}/config.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    for k,v in params.items():
        args[k] = v

    run(config)


