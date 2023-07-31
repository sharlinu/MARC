import argparse
import torch
import time
import imageio
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import RelationalSAC
import os
import json
import sys
sys.path.insert(0, '/Users/sharlinu/Desktop/github/MABoxWorld/environments')
sys.path.insert(0, '/Users/sharlinu/Desktop/github/MABoxWorld/Images')
sys.path.insert(0, '/Users/sharlinu/Desktop/github/MABoxWorld/')
# from environments.box import BoxWorldEnv
sys.path.insert(0, '/home/utke_s@WMGDS.WMG.WARWICK.AC.UK/github/MABoxWorld')
sys.path.insert(0, '/home/utke_s@WMGDS.WMG.WARWICK.AC.UK/github/MABoxWorld/environments')

from environments.box import BoxWorldEnv
from lbforaging.foraging import ForagingEnv
import numpy as np
from enum import Enum
import yaml
from lbforaging.foraging import ForagingEnv
class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

def run(config):
    model_path = config.model_path

    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    gif_path = '{}/{}'.format(eval_path, 'gifs')
    os.makedirs(gif_path, exist_ok=True)

    model = RelationalSAC.init_from_save(model_path)
    print(config.benchmark)
    if 'boxworld' in config.env_id:
        from environments.box import BoxWorldEnv
        env = BoxWorldEnv(
            players=config.player,
            field_size=(config.field,config.field),
            num_colours=config.num_colours,
            goal_length=config.goal_length,
            sight=config.field,
            max_episode_steps=config.test_episode_length,
            grid_observation=config.grid_observation,
            simple=config.simple,
            single=config.single,
            deterministic=config.deterministic,
        )
    elif 'lbf' in  config.env_id:
        env = ForagingEnv(
            players=config.player,
            max_player_level=config.max_player_level,
            field_size=(config.field, config.field),
            max_food=config.max_food,
            grid_observation=config.grid_observation,
            sight=config.field,
            max_episode_steps=config.test_episode_length,
            force_coop=config.force_coop,
            keep_food = config.keep_food,
            simple=config.simple,
        )
    else:
        raise ValueError(f'Cannot cater for the environment {config.env_id}')
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

        from utils.rel_wrapper2 import AbsoluteVKBWrapper
        env = AbsoluteVKBWrapper(env, config.dense)

        obs = env.reset()
        if render:
            env.render()

        for t_i in range(config.test_episode_length):
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
            if render:
                env.render()
                time.sleep(0.5)
            collect_item['l_infos'].append(infos)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if render and (elapsed < ifi):
                time.sleep(ifi - elapsed)
            ep_rew += sum(rewards)

            if all(dones):
                collect_item['finished'] = 1
                break


        l_ep_rew.append(ep_rew)
        print("Reward: {}".format(ep_rew))

        collect_data[ep_i] = collect_item

        with open('{}/collected_data.json'.format(eval_path), 'w') as outfile:
            json.dump(collect_data, outfile,indent=4)

    print("Average reward: {}".format(sum(l_ep_rew)/config.test_n_episodes))
    env.close()


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

