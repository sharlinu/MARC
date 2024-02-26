import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pygame
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
        directory = f'plots/episode_{ep_i}'
        os.makedirs(directory, exist_ok=True)
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
        q_values_a = []
        q_values_b = []
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
            q_a,q_b = model.critic_embeds(obs, torch_actions)
            q_values_a.append(float(q_a.squeeze().detach().numpy()))
            q_values_b.append(float(q_b.squeeze().detach().numpy()))
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
                env.render(save=config.save, name=f"{directory}/step_{t_i}.png")
                time.sleep(0.1)
            # if config.save:
            #     env.save(f"{directory}/step_{t_i}.png")
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
        graph_plots_a = torch.vstack([graph_embeddings[i][0] for i in range(len(graph_embeddings))])
        graph_plots_b = torch.vstack([graph_embeddings[i][1] for i in range(len(graph_embeddings))])
        # temp2 = torch.vstack([node_embeddings[i][1] for i in range(len(node_embeddings))])
        # batch = [node_embeddings[i][0].shape[0] for i in range(len(node_embeddings))]
        batch = [i for i, n in enumerate(node_embeddings) for _ in range(len(n[0]))]
        # activation = torch.vstack([temp1, temp2]).detach().numpy()
        activation = temp1.detach().numpy()
        # graph_activation = graph_plots.detach().numpy()
        graph_activation = torch.vstack([graph_plots_a, graph_plots_b]).detach().numpy()
        graph_labels = [i for i in range(len(graph_plots_a))]
        graph_labels = 2 * graph_labels
        # labels = torch.vstack([labels_1,labels_2])
        labels = labels_1
        unique_features, inverse_indices = torch.unique(labels, dim=0, return_inverse=True)

        # Create a dictionary to map each unique feature vector to a label
        label_dict = {tuple(feature.tolist()): str(feature) for feature in unique_features}
        # colour_dict = {tuple(feature.tolist()): colour for colour, feature in enumerate(unique_features)}
        label_dict[(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] = 'goal'
        label_dict[(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)] = 'object'
        label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper'
        label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 1.0)] = 'id_picker'
        label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper'
        label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)] = 'dropper+obj'
        label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 0.0)] = 'id_dropper+obj'
        label_dict[(1.0, 1.0, 0.0, 1.0, -1.0, 0.0)] = 'id_dropper+iobj'
        label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)] = 'id_dropper'
        label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)] = 'picker'
        label_dict[(1.0, 1.0, 1.0, 1.0, -1.0, 0.0)] = 'id_dropper+obj+goal'
        label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 1.0)] = 'id_picker+obj'
        label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 1.0)] = 'picker+obj'
        label_dict[(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)] = 'picker+obj+goal'
        label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 1.0)] = 'picker+obj'
        label_dict[(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)] = 'full_goal'
        label_dict[(1.0, 1.0, 1.0, 0.0, -1.0, 0.0)] = 'dropper+obj+goal'
        label_dict[(1.0, 0.0, 1.0, 1.0, -1.0, 1.0)] = 'id_picker+goal'
        label_dict[(1.0, 0.0, 1.0, 0.0, -1.0, 0.0)] = 'dropper+goal'
        # Create label vector
        fin_labels = ([label_dict[tuple(feature.tolist())] for feature in labels])

        print(activation.shape)
        tsne_model = TSNE(n_components=3, perplexity=5)
        # input needs to be (n_samples, n_features)
        d = tsne_model.fit_transform(activation)
        d_graph = tsne_model.fit_transform(graph_activation)

        df = pd.DataFrame(d, columns=['x', 'y', 'z'])
        df_graph = pd.DataFrame(d_graph, columns=['x', 'y', 'z'])
        df_graph['steps'] = graph_labels
        q_values = q_values_a + q_values_b
        agents = [0 for _ in range(len(q_values_a))] + [1 for _ in range(len(q_values_b))]
        df_graph['q_values'] = q_values
        df_graph['agent'] = agents
        df['label'] = fin_labels
        df['batch'] = batch
        # frames = []
        # fig = go.Figure(
        #     data=[go.Scatter3d(x=[], y=[], z=[],
        #                        mode="markers",
        #                        marker=dict(color="red",
        #                                    size=10))])

        # for b in set(batch):
        #     frames.append(go.Frame(data=go.Scatter3d(
        #         x=df[df['batch'] == b]['x'],
        #         y=df[df['batch'] == b]['y'],
        #         z=df[df['batch'] == b]['z'],
        #         name=df[df['batch'] == b]['label'],
        #     ),
        #         traces=[0],
        #         name=f'frame{b}'
        #         ))
        # fig.update(frames=frames)
        # fig.update_layout(updatemenus=[dict(type="buttons",
        #                                     buttons=[dict(label="Play",
        #                                                   method="animate",
        #                                                   args=[None, dict(frame=dict(redraw=True, fromcurrent=True,
        #                                                                               mode='immediate'))])])])
        # fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')
        min_ax = min(min(df['x']), min(df['y']), min(df['z']))
        max_ax = max(max(df['x']), max(df['y']), max(df['z']))
        fig = px.scatter_3d(df,
                            x='x',
                            y='y',
                            z='z',
                            animation_frame='batch',
                            color='label',
                            range_x=[min_ax, max_ax],
                            range_y=[min_ax, max_ax],
                            range_z=[min_ax, max_ax],
                            )

        fig.update_scenes(aspectmode='cube')
        fig.show()
        fig.write_html(f"plots/episode_{ep_i}/node_embeddings.html")

        fig_full = px.scatter_3d(df,
                            x='x',
                            y='y',
                            z='z',
                            # animation_frame='batch',
                            color='label',
                            # range_x=[min_ax, max_ax],
                            # range_y=[min_ax, max_ax],
                            # range_z=[min_ax, max_ax],
                            )
        fig_full.show()
        fig_full.write_html(f"plots/episode_{ep_i}/full_node_embeddings.html")

        fig_graph = px.scatter_3d(df_graph,
                            x='x',
                            y='y',
                            z='z',
                            color='q_values',
                            hover_name="q_values",
                            hover_data=
                              {'x': False,
                               'y': False,
                               'z': False,
                               'agent': True,
                               'q_values': False,
                               'steps': True}
                            )
        fig_graph.show()
        fig_graph.write_html(f"plots/episode_{ep_i}/q_embeddings.html")

        fig_graph = px.scatter_3d(df_graph,
                            x='x',
                            y='y',
                            z='z',
                            color='steps',
                            hover_name="steps",
                            hover_data=
                                  {'x':False,
                                   'y': False,
                                   'z': False,
                                   'agent':True,
                                   'q_values':True,
                                   'steps':False}
                            )
        fig_graph.show()
        fig_graph.write_html(f"plots/episode_{ep_i}/graph_embeddings.html")

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
    parser.add_argument("--render", default=True, action="store_true",
                        help="render")
    parser.add_argument("--save", default=True, action="store_true",
                        help="save step images")
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
