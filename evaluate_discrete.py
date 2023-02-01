import argparse
import torch
import time
import imageio
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import AttentionSAC
import os
import json
from box import BoxWorldEnv
import numpy as np

def run(config):
    model_path = config.model_path
    # if config.incremental is not None:
    #     model_path = model_path / 'incremental' / ('model_ep%i.pt' %
    #                                                config.incremental)
    # else:
    #     model_path = model_path / 'model.pt'

    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    gif_path = '{}/{}'.format(eval_path, 'gifs')
    os.makedirs(gif_path, exist_ok=True)

    maddpg = AttentionSAC.init_from_save(model_path)
    print(config.benchmark)
    env = BoxWorldEnv(
        players=2,
        field_size=(5,5),
        num_colours=2,
        goal_length=2,
        sight=5,
        max_episode_steps=200,
        simple=False,
        relational=False,
        deterministic=True,
    )

    maddpg.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    collect_data = {}
    l_ep_rew = []
    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))

        frames = []
        collect_item = {
            'ep': ep_i,
            'final_reward': 0,
            'l_infos': [],
            'l_rewards': []
        }
        l_rewards = []
        ep_rew = 0

        obs = env.reset()

        if config.save_gifs:
            frames = []
            frames.append(env.render('rgb_array')[0])

        for t_i in range(config.episode_length):
            calc_start = time.time()

            if config.no_render != False:
                frames.append(env.render(mode='rgb_array', close=False)[0])  

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            env.render()
            time.sleep(0.5)
            collect_item['final_reward'] = sum(rewards)
            collect_item['l_rewards'].append(sum(rewards))
            collect_item['l_infos'].append(infos)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < ifi:
                time.sleep(ifi - elapsed)
            ep_rew += sum(rewards)

            if all(dones):
                collect_item['finished'] = 1
                break

        if config.save_gifs:
            print('{}/{}.gif'.format(gif_path, ep_i))
            imageio.mimsave('{}/{}.gif'.format(gif_path, ep_i),
                            frames, duration=ifi)
        
        l_ep_rew.append(ep_rew)
        print("Reward: {}".format(ep_rew))

        collect_data[ep_i] = collect_item

        with open('{}/collected_data.json'.format(eval_path), 'w') as outfile:
            json.dump(collect_data, outfile,indent=4)
        #     if config.save_gifs:
        #         frames.append(env.render('rgb_array')[0])
        #     calc_end = time.time()
        #     elapsed = calc_end - calc_start
        #     if elapsed < ifi:
        #         time.sleep(ifi - elapsed)
        #     env.render('human')
        # if config.save_gifs:
        #     gif_num = 0
        #     while (gif_path / ('%i_%i.gif' % (gif_num, ep_i))).exists():
        #         gif_num += 1
        #     imageio.mimsave(str(gif_path / ('%i_%i.gif' % (gif_num, ep_i))),
        #                     frames, duration=ifi)
    print("Average reward: {}".format(sum(l_ep_rew)/config.n_episodes))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="model_path")
    # parser.add_argument("model_name",
    #                     help="Name of model")
    # parser.add_argument("run_num", default=1, type=int)
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=10, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--no_render", action="store_false",
                        help="render")
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    config = parser.parse_args()

    run(config)