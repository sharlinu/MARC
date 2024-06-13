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
from algorithms.attention_sac import RelationalSAC, AttentionSAC
import os
import gym
import macpp
from utils.env_wrappers import GridObs, FlatObs
from sklearn.cluster import KMeans

from PIL import Image
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import dash
from dash.exceptions import PreventUpdate
from dash import dcc, html
from dash.dependencies import Input, Output
hook_activation = {}
def get_activation(name):
    def hook(model, input, output):
        hook_activation[name] = output.detach()
    return hook

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
    print(f'Loading model from {model_path}')
    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    config.device = 'cpu'
    model, _ = AttentionSAC.init_from_save(model_path, device=config.device)
    model.critic.critics[0].critic_nl.register_forward_hook(get_activation('critic_nl'))

    env = gym.make(f"macpp-{config.field}x{config.field}-{config.player}a-{config.pp['n_picker']}p-{config.pp['n_objects']}o-v3", debug_mode=False)
    env = FlatObs(env)

    model.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    l_ep_rew = []
    df_full = pd.DataFrame()
    # linear_embedds = torch.empty((0,128))
    linear_embeddings = []

    for ep_i in range(config.eval_n_episodes):
        directory = f'plots/pp/episode_{ep_i}'
        os.makedirs(directory, exist_ok=True)
        state_label = []
        images = []
        print("Episode %i of %i" % (ep_i + 1, config.eval_n_episodes))
        ep_rew = 0
        obs = env.reset()

        q_values_a = []
        q_values_b = []
        if config.render:
            env.render()
        label = 0
        rewards = [-0.1, -0.1]
        for t_i in range(config.eval_episode_length):
            if config.render:
                img = f"{directory}/step_{t_i}.png"
                env.render(save=config.save, name=img)
                images.append(Image.open(img))
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(model.n_agents)]
            # get actions as torch Variables
            torch_actions = model.step(torch_obs, explore=False)
            q_a, q_b = model.critic_embeds(torch_obs, torch_actions)
            q_values_a.append(float(q_a.squeeze().detach().numpy()))
            q_values_b.append(float(q_b.squeeze().detach().numpy()))
            linear_embeddings.append(hook_activation['critic_nl'])

            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
            if sum(rewards) == -0.2:
                pass
            elif rewards[0] == 0.4:
                label +=1
            elif sum(rewards) == 0.8:
                label += 1
            elif sum(rewards) >= 2.8:
                label += 1
            elif sum(rewards) == 3.8:
                label += 1
            else:
                print(f'rewards are not categorised with  {rewards}')
                time.sleep(10)
            state_label.append(label)
            # print(state_label)
            # print('actions', actions)
            obs, rewards, dones, infos = env.step(actions)
            env.render()
            # if config.save:
            #     env.save(f"{directory}/step_{t_i}.png")
            ep_rew += sum(rewards)
            if all(dones):
                break



        if t_i>23:
            success = 0
            print('using embeddings even though stuck')
        else:
            success = 1
        print('using embeddings')
        l_ep_rew.append(ep_rew)

        # temp2 = torch.vstack([node_embeddings[i][1] for i in range(len(node_embeddings))])
        # batch = [node_embeddings[i][0].shape[0] for i in range(len(node_embeddings))]
        # steps = [i for i, n in enumerate(node_embeddings) for _ in range(len(n[0]))]
        # batch_images = [images[i] for i, n in enumerate(linear_embeddings) for _ in range(len(n[0]))]

        # activation = torch.vstack([temp1, temp2]).detach().numpy()
        # activation = temp1.detach().numpy()


        df = pd.DataFrame()
        q_values = q_values_a #+ q_values_b
        agents = [0 for _ in range(len(q_values_a))] # + [1 for _ in range(len(q_values_b))]
        df['images'] = images
        df['q_values'] = q_values
        df['episode'] = ep_i
        df['success'] = success
        df['state_label']  = state_label
        df_full = pd.concat([df_full, df])

    print("Average reward: {}".format(round(sum(l_ep_rew)/len(l_ep_rew)),3))
    env.close()

    model = umap.UMAP(n_components=3)

    # input needs to be (n_samples, n_features)
    # d = tsne_model.fit_transform(activation)
    activation = torch.vstack([i for i in linear_embeddings])

    kmeans_model = KMeans(n_clusters=3, random_state=0)
    kmeans_model = kmeans_model.fit(activation.detach().numpy())

    pred_labels = kmeans_model.predict(activation.detach().numpy())

    d = model.fit_transform(activation.detach().numpy())

    df_full['x'] = d[:, 0]
    df_full['y'] = d[:, 1]
    df_full['z'] = d[:, 2]
    df_full['k_label'] = pred_labels


    app = dash.Dash(__name__)

    fig_full = px.scatter_3d(df_full,
                             x='x',
                             y='y',
                             z='z',
                             color='k_label',
                             hover_data=
                             {'x': False,
                              'y': False,
                              'z': False,
                              'state_label': True,
                              'success': True,
                              'q_values': True,
                              'episode': True},
                             custom_data = ["images"]
                             )
    # fig_full.write_html(f"plots/pp/all_node_embeddings.html")


    app.layout = html.Div(
        [
            dcc.Graph(
                id="graph",
                figure=fig_full,
                style={'display': 'inline-block', 'width': '100vh'}
            ),
            html.Img(id='image', src='',style={'display':'inline-block', 'width': '80vh'})
        ]

    )

    @app.callback(
        Output('image', 'src'),
        Input('graph', 'hoverData'))
    def open_url(hoverData):
        if hoverData:
            return hoverData["points"][0]["customdata"][0]
        else:
            raise PreventUpdate


    return app



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        # default='experiments/MARC/pp/2024-01-24_pp_10x10_2a_1p_2o-v3_b0_std_seed4001/saved_models/ckpt_final.pth.tar',
                        default = 'experiments/MARC/pp/2024-03-12_pp_5x5_2a_1p_1o-v3_GAT_std_seed4001/saved_models/ckpt_best_avg_r1.6320000886917114.pth.tar',
                        help="model_path")
    parser.add_argument("--eval_n_episodes", default=30, type=int)
    parser.add_argument("--eval_episode_length", default=40, type=int)
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

    app = run(config)
    app.run_server(debug=False)

