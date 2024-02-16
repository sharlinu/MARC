import argparse
import torch
import time
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import RelationalSAC
import os
import gym
import macpp
from utils.env_wrappers import GridObs
from utils.rel_wrapper2 import AbsoluteVKBWrapper
import numpy as np
import yaml


def run(config):
    model_path = config.model_path

    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    config.device = 'cpu'
    model, _ = RelationalSAC.init_from_save(model_path, device=config.device)

    env=gym.make(f"macpp-10x10-2a-1p-2o-v3", debug_mode=False)
    env = GridObs(env)
    attr_mapping = config.pp['attr_mapping']
    env = AbsoluteVKBWrapper(env,
                             attr_mapping=attr_mapping,
                             dense=config.marc['dense'],
                             background_id=config.marc['background_id'],
                             abs_id=config.marc['abs_id']
                             )
    model.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    l_ep_rew = []
    node_embeddings = []
    node_concepts = []
    graph_embeddings = []

    for ep_i in range(config.eval_n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.eval_n_episodes))

        ep_rew = 0


        obs = env.reset()
        if config.render:
            env.render()

        for t_i in range(config.eval_episode_length):
            calc_start = time.time()

            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
            torch_obs = [Variable(torch.Tensor(torch_obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(model.n_agents)]
            # get actions as torch Variables
            torch_actions = model.step(torch_obs,explore=False)
            model.critic_embeds(obs, torch_actions)
            node_embeddings.append(model.critic.node_embeddings)
            node_concepts.append(model.critic.node_concepts)
            graph_embeddings.append(model.critic.graph_embeddings)
            # convert actions to numpy arrays

            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]

            # print('actions', actions)
            obs, rewards, dones, infos = env.step(actions)
            if config.render:
                env.render()
                time.sleep(0.5)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if config.render and (elapsed < ifi):
                time.sleep(ifi - elapsed)
            ep_rew += sum(rewards)

            if all(dones):
                break


        l_ep_rew.append(ep_rew)
        # print("Reward: {}".format(ep_rew))
    print(node_embeddings)
    print("Average reward: {}".format(sum(l_ep_rew)/config.eval_n_episodes))
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default='experiments/MARC/pp/2024-01-24_pp_10x10_2a_1p_2o-v3_b0_std_seed4001/saved_models/ckpt_final.pth.tar',
                        help="model_path")
    parser.add_argument("--eval_n_episodes", default=1, type=int)
    parser.add_argument("--eval_episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render", default=False, action="store_true",
                        help="render")
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    config = parser.parse_args()
    args = vars(config)
    eval_path = Path(config.model_path)
    dir_exp = Path(*eval_path.parts[:-2])
    with open(f"{dir_exp}/config.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    for k,v in params.items():
        args[k] = v

    run(config)

