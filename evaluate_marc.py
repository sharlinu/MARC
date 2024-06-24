import argparse
import torch
import time
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import RelationalSAC
import os
import json
import sys
import gym
import numpy as np
from enum import Enum
import yaml
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
    config.device = 'cpu'
    model, _ = RelationalSAC.init_from_save(model_path, device=config.device)
    if 'boxworld' in config.env_id:
        from environments.box import BoxWorldEnv
        env = BoxWorldEnv(
            players=config.player,
            field_size=(config.field,config.field),
            num_colours=config.num_colours,
            goal_length=config.goal_length,
            sight=config.field,
            max_episode_steps=config.episode_length,
            grid_observation=config.grid_observation,
            simple=config.simple,
            single=config.single,
            deterministic=config.deterministic,
        )
    elif 'lbf' in  config.env_id:
        env = ForagingEnv(
            players=config.player,
            max_player_level=config.lbf['max_player_level'],
            # max_player_level=2,
            field_size=(config.field, config.field),
            max_food=config.lbf['max_food'],
            grid_observation=config.grid_observation,
            sight=config.field,
            max_episode_steps=config.episode_length,
            force_coop=config.lbf['force_coop'],
            keep_food = config.lbf['keep_food'],
            # simple=config.simple,
        )
        attr_mapping = config.lbf['attr_mapping']
    elif 'bpush' in  config.env_id:
        from bpush.environment import BoulderPush
        env = BoulderPush(
            height=config.field,
            width=config.field,
            n_agents=config.player,
            sensor_range=config.bpush['sensory_range'],
        )
        attr_mapping = config.bpush['attr_mapping']
    elif 'wolf' in config.env_id:
        from Wolfpack_gym.envs.wolfpack import Wolfpack
        env = Wolfpack(
            grid_height=config.field,
            grid_width=config.field,
            num_players=config.player,
            max_food_num=config.wolfpack['max_food_num'],
            obs_type='grid',
            close_penalty=config.wolfpack['close_penalty'],
            sparse=config.wolfpack['sparse'],
        )
        attr_mapping = config.wolfpack['attr_mapping']
    elif 'pp' in config.env_id:
        import macpp
        from utils.env_wrappers import GridObs
        env=gym.make(f"macpp-{config.field}x{config.field}-{config.player}a-{config.pp['n_picker']}p-{config.pp['n_objects']}o-{config.pp['version']}",
                       debug_mode=False)
        env = GridObs(env)
        attr_mapping = config.pp['attr_mapping']
    elif 'MAPE' in config.env_id:
        from multiagent.MPE_env import MPEEnv, GraphMPEEnv
        env = GraphMPEEnv(config)
    else:
        raise ValueError(f'Cannot cater for the environment {config.env_id}')

    from utils.rel_wrapper2 import AbsoluteVKBWrapper
    if 'MAPE' in config.env_id:
        config.grid_observation = False
        # env.grid_observation = False
        env.agents = [None] * len(env.action_space)
        env.n_agents = len(env.action_space)
        env.n_rel_rules = 5
        obs, agent_id, node_obs, adj = env.reset()

        # env.n_attr = node_obs.shape[3]

    else:
        env = AbsoluteVKBWrapper(env,
                                 attr_mapping=attr_mapping,
                                 dense=config.marc['dense'],
                                 background_id=config.marc['background_id'],
                                 abs_id=config.marc['abs_id']
                                 )
    model.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    collect_data = {}
    l_ep_rew = []

    for ep_i in range(config.eval_n_episodes):
    # for ep_i in range(20):
        print("Episode %i of %i" % (ep_i + 1, config.eval_n_episodes))

        collect_item = {
            'ep': ep_i,
            'final_reward': 0,
            'l_infos': [],
            'l_rewards': [],
            'agent_collisions': 0,
            'obst_collisions': 0,
        }
        l_rewards = []
        ep_rew = 0

        if 'MAPE' in config.env_id:
            obs, agent_id, node_obs, adj = env.reset()
        else:
            obs = env.reset()
        if config.render:
            env.render()

        for t_i in range(config.eval_episode_length):
            calc_start = time.time()

            # rearrange observations to be per agent, and convert to torch Variable
            if config.grid_observation:
                obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
            # if 'MAPE' in config.env_id:
            #     torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                           requires_grad=False)
            #                  for i in range(model.n_agents)]
            # else:
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(model.n_agents)]
            # get actions as torch Variables
            torch_actions = model.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
            # print('actions', actions)
            if 'MAPE' in config.env_id:
                actions = [ac.data.numpy().flatten() for ac in torch_actions]
                obs, agent_id, node_obs, adj, rewards, dones, infos = env.step(actions)
            else:
                obs, rewards, dones, infos = env.step(actions)

            if config.render:
                if 'lbf' in config.env_id:
                    env.render(actions=actions)
                else:
                    env.render()
                time.sleep(0.5)
            collect_item['l_infos'].append(infos)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if config.render and (elapsed < ifi):
                time.sleep(ifi - elapsed)
            ep_rew += sum(rewards)
            # print('rewards', rewards, ep_rew)
            l_rewards.append(sum(rewards))
            if all(dones):
                collect_item['finished'] = 1
                break

        # print(f"obst coll: {sum([d['Num_obst_collisions'] for d in infos])}, a coll: {sum([d['Num_agent_collisions'] for d in infos])}")
        l_ep_rew.append(ep_rew)
        # print("Reward: {} for {} steps".format(ep_rew, t_i))
        collect_item['l_rewards'] = l_rewards
        collect_item['obst_collisions'] = sum([d['Num_obst_collisions'] for d in infos])
        collect_item['agent_collisions'] = sum([d['Num_agent_collisions'] for d in infos])
        collect_data[ep_i] = collect_item
        collect_item['final_reward'] = np.sum(l_rewards)

        with open('{}/collected_data.json'.format(eval_path), 'w') as outfile:
           # print(f'dumped eval dict at {eval_path}')
            json.dump(collect_data, outfile,indent=4)

    print("Average reward: {}".format(sum(l_ep_rew)/config.eval_n_episodes))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="model_path")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--eval_n_episodes", default=10, type=int)
    parser.add_argument("--eval_episode_length", default=50, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render", default=False, action="store_true",
                        help="render")
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    parser.add_argument("--save", action="store_true",
                        help="to save visualisation of the environment")

    config = parser.parse_args()
    args = vars(config)
    eval_path = Path(config.model_path)
    dir_exp = Path(*eval_path.parts[:-2])
    with open(f"{dir_exp}/config.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    for k,v in params.items():
        args[k] = v
    # args['field'] = 15
    run(config)

