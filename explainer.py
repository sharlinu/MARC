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
import matplotlib.pyplot as plt
import torch.nn.functional as Fun

from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances_argmin_minl

def plot_activation_space(data):
    # rows = len(data)
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_title(f" Activations of Layer {layer_num} {note}")

    scatter = ax.scatter(data[:,0], data[:,1], cmap='rainbow')
    ax.legend(handles=scatter.legend_elements()[0], bbox_to_anchor=(1.05, 1))

    # plt.savefig(os.path.join(path, f"{layer_num}_layer{naming_help}.png"))
    plt.show()

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
    embeddings = ['node_embeddings', 'node_concepts', 'graph_embeddings']

    for ep_i in range(config.eval_n_episodes):
        node_embeddings = []
        node_concepts = []
        graph_embeddings = []
        print("Episode %i of %i" % (ep_i + 1, config.eval_n_episodes))

        ep_rew = 0


        obs = env.reset()

        labels_1 = torch.empty((0,6))
        labels_2 = torch.empty((0,6))
        # attr_mapping: {'agent': 0,
        #              'objects': 1,
        #              'goals': 2,
        #              'id':3,
        #              'carrying':4,
        #              'picker':5,}

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
            labels_1 = torch.cat([labels_1, obs[0]['unary_tensor']])
            labels_2 = torch.cat([labels_2, obs[1]['unary_tensor']])
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
        # # print("Reward: {}".format(ep_rew))
        #
        # for embedding in embeddings:
        #     directory = f'episode{ep_i}/{embedding}_agent0'
        #     if os.path.exists(directory):
        #         os.remove(directory)
        #     os.makedirs(directory)
        #     for idx, tensor in enumerate(eval(embedding)):
        #         torch.save(tensor[0], f"{directory}/tensor{idx}.pt")

        temp1 = torch.vstack([node_embeddings[i][0] for i in range(len(node_embeddings))])
        # temp2 = torch.vstack([node_embeddings[i][1] for i in range(len(node_embeddings))])
        # activation = torch.vstack([temp1, temp2]).detach().numpy()
        activation = temp1.detach().numpy()
        # labels = torch.vstack([labels_1,labels_2])
        labels = labels_1
        unique_features, inverse_indices = torch.unique(labels, dim=0, return_inverse=True)

        # Create a dictionary to map each unique feature vector to a label
        label_dict = {tuple(feature.tolist()): str(feature) for feature in unique_features}
        colour_dict = {tuple(feature.tolist()): colour for colour, feature in enumerate(unique_features)}
        label_dict[(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] = 'goal'
        label_dict[(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)] = 'object'
        label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper'
        label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 1.0)] = 'id_picker'
        label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper'
        label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)] = 'c_dropper'
        label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 0.0)] = 'id_c_dropper'
        label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 1.0)] = 'id_c_picker'
        label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 1.0)] = 'c_picker'
        label_dict[(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)] = 'full_goal'
        label_dict[(1.0, 1.0, 1.0, 0.0, -1.0, 0.0)] = 'full_goal'
        label_dict[(1.0, 0.0, 1.0, 1.0, -1.0, 1.0)] = 'id_picker_a_goal'
        label_dict[(1.0, 0.0, 1.0, 0.0, -1.0, 0.0)] = 'dropper_a_goal'
        # Create label vector
        fin_labels = ([label_dict[tuple(feature.tolist())] for feature in labels])
        fin_colours = ([colour_dict[tuple(feature.tolist())] for feature in labels])

        print(activation.shape)
        tsne_model = TSNE(n_components=2, perplexity=5)
        # input needs to be (n_samples, n_features)
        d = tsne_model.fit_transform(activation)
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(d[:, 0], d[:, 1], c=fin_colours, cmap='rainbow')
        ax.legend(handles=scatter.legend_elements()[0], labels= set(fin_labels)) # bbox_to_anchor=(0.4, 0.85)
        # plt.savefig(f"episode{ep_i}/node_embeddings.png")
        plt.savefig(f"node_embeddings.png")
        plt.show()

    print("Average reward: {}".format(round(sum(l_ep_rew)/config.eval_n_episodes),2))
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
