import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from pathlib import Path
from torch.autograd import Variable
from algorithms.attention_sac import RelationalSAC
import os
import gym
import macpp
from utils.env_wrappers import GridObs
from utils.rel_wrapper2 import AbsoluteVKBWrapper, GATWrapper
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
    # plt.show()

def run(config):
    model_path = config.model_path
    print(f'Loading model from {model_path}')
    # create folder for evaluating
    eval_path = Path(config.model_path)
    eval_path = Path(*eval_path.parts[:-2])
    eval_path = '{}/evaluate'.format(eval_path)
    os.makedirs(eval_path, exist_ok=True)

    config.device = 'cpu'
    model, _ = RelationalSAC.init_from_save(model_path, device=config.device, load_critic=True)
    model.critic.critics_head[0].critic_nl.register_forward_hook(get_activation('critic_nl'))
    env = gym.make(f"macpp-{config.field}x{config.field}-{config.player}a-{config.pp['n_picker']}p-{config.pp['n_picker']}o-v3",
                   debug_mode=False,
                   # render_mode='human'
                   )
    env = GridObs(env)
    attr_mapping = config.pp['attr_mapping']
    if config.marc['graph_layer'] == 'GAT':
        env = GATWrapper(env=env,
                         attr_mapping=attr_mapping,
                         dense=config.marc['dense'],
                         )
    elif config.marc['graph_layer'] =='RGCN':
        env = AbsoluteVKBWrapper(env,
                                 attr_mapping=attr_mapping,
                                 dense=config.marc['dense'],
                                 background_id=config.marc['background_id'],
                                 abs_id=config.marc['abs_id']
                                 )
    else:
        ValueError

    model.prep_rollouts(device='cpu')
    ifi = 1 / config.fps  # inter-frame interval
    l_ep_rew = []
    embeddings = ['node_embeddings', 'node_concepts', 'graph_embeddings']
    df_full = pd.DataFrame()
    df_full_graph = pd.DataFrame()
    try:
        embed_size = config.marc['embed_size']
    except KeyError:
        embed_size = 128
    graph_embedds = torch.empty((0, embed_size))
    node_embedds = torch.empty((0, embed_size))
    linear_embedds = torch.empty((0, 128))
    for ep_i in range(config.eval_n_episodes):
        directory = f'plots/pp/episode_{ep_i}'
        os.makedirs(directory, exist_ok=True)
        node_embeddings = []
        node_concepts = []
        graph_embeddings = []
        linear_embeddings = []
        state_label = []
        images = []
        print("Episode %i of %i" % (ep_i + 1, config.eval_n_episodes))
        ep_rew = 0
        obs = env.reset(seed=ep_i)

        if config.marc['graph_layer'] == 'GAT':
            labels_1 = torch.empty((0,8))
            labels_2 = torch.empty((0,8))
        else:
            labels_1 = torch.empty((0,6))
            labels_2 = torch.empty((0,6))

        q_values_a = []
        q_values_b = []
        if config.render:
            env.render()
        label = 0
        rewards = [-0.1,-0.1]
        for t_i in range(config.eval_episode_length):

            if config.render:
                img = f"{directory}/step_{t_i}.png"
                env.render(save=config.save, name=img)
                images.append(Image.open(img))
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
            linear_embeddings.append(hook_activation['critic_nl'])

            labels_1 = torch.cat([labels_1, obs[0]['unary_tensor']])
            labels_2 = torch.cat([labels_2, obs[1]['unary_tensor']])
            # convert actions to numpy arrays

            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]

            # print('actions', actions)
            import time
            # time.sleep(1)
            # print(rewards)
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
            obs, rewards, dones, infos = env.step(actions)

            # # import time.sleep(1)
            # if sum(rewards) == -0.2:
            #     state_label.append(0)
            # elif rewards[0] == 0.4:
            #     state_label.append(2)
            # elif sum(rewards) == 0.8:
            #     state_label.append(1)
            # elif sum(rewards) == 2.8:
            #     state_label.append(3)
            # else:
            #     print(f'rewards are not categorised with  {rewards}')
            # print(state_label)

            # if config.save:
            #     env.save(f"{directory}/step_{t_i}.png")
            ep_rew += sum(rewards)
            if all(dones):
                break


        # print(rewards, t_i)
        if t_i>23:
        #     continue
            success = 0
            print('using graph embeddings even though stuck')
        else:
            success = 1

        l_ep_rew.append(ep_rew)
        temp1 = torch.vstack([node_embeddings[i][0] for i in range(len(node_embeddings))])
        graph_plots_a = torch.vstack([graph_embeddings[i][0] for i in range(len(graph_embeddings))])
        # graph_plots_b = torch.vstack([graph_embeddings[i][1] for i in range(len(graph_embeddings))])
        linear = torch.vstack([linear_embeddings[i][0] for i in range(len(linear_embeddings))])
        graph_embedds  = torch.cat([graph_embedds, graph_plots_a])
        linear_embedds = torch.cat([linear_embedds, linear])
        node_embedds = torch.cat([node_embedds, temp1])
        # temp2 = torch.vstack([node_embeddings[i][1] for i in range(len(node_embeddings))])
        # batch = [node_embeddings[i][0].shape[0] for i in range(len(node_embeddings))]
        steps = [i for i, n in enumerate(node_embeddings) for _ in range(len(n[0]))]
        batch_images = [images[i] for i, n in enumerate(node_embeddings) for _ in range(len(n[0]))]

        # activation = torch.vstack([temp1, temp2]).detach().numpy()
        activation = temp1.detach().numpy()
        graph_activation = graph_plots_a.detach().numpy()
        # graph_activation = torch.vstack([graph_plots_a, graph_plots_b]).detach().numpy()
        graph_labels = [i for i in range(len(graph_plots_a))]
        graph_labels =  graph_labels
        # labels = torch.vstack([labels_1,labels_2])
        labels = labels_1
        unique_features, inverse_indices = torch.unique(labels, dim=0, return_inverse=True)

        # Create a dictionary to map each unique feature vector to a label
        label_dict = {tuple(feature.tolist()): str(feature) for feature in unique_features}
        # colour_dict = {tuple(feature.tolist()): colour for colour, feature in enumerate(unique_features)}
        if config.marc['graph_layer'] == 'GAT':
            for i in range(config.field):
                for j in range(config.field):
                    label_dict[(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, i, j)] = 'goal'
                    label_dict[(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, i, j)] = 'object'
                    label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 0.0,i,j)] = 'dropper'
                    label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 1.0,i,j)] = 'id_picker'
                    label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 0.0,i,j)] = 'dropper+obj'
                    label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 0.0,i,j)] = 'id_dropper+obj'
                    label_dict[(1.0, 1.0, 0.0, 1.0, -1.0, 0.0,i,j)] = 'id_dropper+iobj'
                    label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 0.0,i,j)] = 'id_dropper'
                    label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 1.0,i,j)] = 'picker'
                    label_dict[(1.0, 1.0, 1.0, 1.0, -1.0, 0.0,i,j)] = 'id_dropper+obj+goal'
                    label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 1.0,i,j)] = 'id_picker+obj'
                    label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 1.0,i,j)] = 'picker+obj'
                    label_dict[(1.0, 1.0, 1.0, 0.0, 1.0, 1.0,i,j)] = 'picker+obj+goal'
                    label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 1.0,i,j)] = 'picker+obj'
                    label_dict[(0.0, 1.0, 1.0, 0.0, 0.0, 0.0,i,j)] = 'full_goal'
                    label_dict[(1.0, 1.0, 1.0, 0.0, -1.0, 0.0,i,j)] = 'dropper+obj+goal'
                    label_dict[(1.0, 0.0, 1.0, 1.0, -1.0, 1.0,i,j)] = 'id_picker+goal'
                    label_dict[(1.0, 0.0, 1.0, 0.0, -1.0, 0.0,i,j)] = 'dropper+goal'
        else:
            label_dict[(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] = 'goal'
            label_dict[(0.0, 1.0, 0.0, 0.0, 0.0, 0.0)] = 'object'
            label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper'
            label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 0.0)] = 'dropper'
            label_dict[(1.0, 0.0, 0.0, 1.0, -1.0, 1.0)] = 'picker'
            label_dict[(1.0, 0.0, 0.0, 0.0, -1.0, 1.0)] = 'picker'
            label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)] = 'dropper+obj'
            label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 0.0)] = 'dropper+obj'
            label_dict[(1.0, 1.0, 0.0, 1.0, -1.0, 0.0)] = 'dropper+iobj'
            label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper+iobj'

            label_dict[(1.0, 1.0, 1.0, 1.0, -1.0, 0.0)] = 'dropper+obj+goal'
            label_dict[(1.0, 1.0, 1.0, 0.0, -1.0, 0.0)] = 'dropper+obj+goal'
            label_dict[(1.0, 1.0, 0.0, 1.0, 1.0, 1.0)] = 'picker+obj'
            label_dict[(1.0, 1.0, 0.0, 0.0, 1.0, 1.0)] = 'picker+obj'
            label_dict[(1.0, 1.0, 1.0, 0.0, 1.0, 1.0)] = 'picker+obj+goal'
            label_dict[(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)] = 'picker+obj+goal'
            label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 1.0)] = 'picker+iobj'
            label_dict[(1.0, 1.0, 0.0, 1.0, -1.0, 1.0)] = 'picker+iobj'
            label_dict[(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)] = 'full_goal'
            label_dict[(1.0, 0.0, 1.0, 1.0, -1.0, 1.0)] = 'picker+goal'
            label_dict[(1.0, 0.0, 1.0, 0.0, -1.0, 1.0)] = 'picker+goal'
            label_dict[(1.0, 0.0, 1.0, 0.0, -1.0, 0.0)] = 'dropper+goal'
            label_dict[(1.0, 0.0, 1.0, 1.0, -1.0, 0.0)] = 'dropper+goal'
            label_dict[(1.0, 1.0, 0.0, 0.0, -1.0, 0.0)] = 'dropper+iobj'
            label_dict[(1.0, 1.0, 0.0, 1.0, -1.0, 0.0)] = 'dropper+iobj'
        # Create label vector
        fin_labels = ([label_dict[tuple(feature.tolist())] for feature in labels])

        print(activation.shape)
        tsne_model = TSNE(n_components=3, perplexity=1)
        # input needs to be (n_samples, n_features)
        d = tsne_model.fit_transform(activation)
        d_graph = tsne_model.fit_transform(graph_activation)

        df = pd.DataFrame(d, columns=['x', 'y', 'z'])
        df_graph = pd.DataFrame(d_graph, columns=['x', 'y', 'z'])
        df_graph['steps'] = graph_labels
        df_graph['success'] = success
        q_values = q_values_a #+ q_values_b
        agents = [0 for _ in range(len(q_values_a))] # + [1 for _ in range(len(q_values_b))]
        df_graph['q_values'] = q_values
        df_graph['agent'] = agents
        df['label'] = fin_labels
        df['steps'] = steps
        df['images'] = batch_images
        df['success'] = success
        df_graph['state'] = state_label
        df_graph['images'] = images


        min_ax = min(min(df['x']), min(df['y']), min(df['z']))
        max_ax = max(max(df['x']), max(df['y']), max(df['z']))
        fig = px.scatter_3d(df,
                            x='x',
                            y='y',
                            z='z',
                            animation_frame='steps',
                            color='label',
                            range_x=[min_ax, max_ax],
                            range_y=[min_ax, max_ax],
                            range_z=[min_ax, max_ax],
                            )

        fig.update_scenes(aspectmode='cube')
        # fig.show()
        fig.write_html(f"plots/pp/episode_{ep_i}/node_embeddings.html")

        fig_full = px.scatter_3d(df,
                                 x='x',
                                 y='y',
                                 z='z',
                                 color='label',
                                 )
        # fig_full.show()
        fig_full.write_html(f"plots/pp/episode_{ep_i}/full_node_embeddings.html")

        fig_graph = px.scatter_3d(df_graph,
                                  x='x',
                                  y='y',
                                  z='z',
                                  color=config.hue,
                                  hover_name="q_values",
                                  hover_data=
                                  {'x': False,
                                   'y': False,
                                   'z': False,
                                   'agent': True,
                                   'q_values': False,
                                   'steps': True}
                                  )
        # fig_graph.show()
        fig_graph.write_html(f"plots/pp/episode_{ep_i}/q_embeddings.html")

        fig_graph = px.scatter_3d(df_graph,
                                  x='x',
                                  y='y',
                                  z='z',
                                  color=config.hue,
                                  hover_name="steps",
                                  hover_data=
                                  {'x':False,
                                   'y': False,
                                   'z': False,
                                   'agent':True,
                                   'q_values':True,
                                   'steps':False}
                                  )
        # fig_graph.show()
        fig_graph.write_html(f"plots/pp/episode_{ep_i}/graph_embeddings.html")
        df['episode'] = ep_i
        df_graph['episode'] = ep_i
        df_full = pd.concat([df_full, df])
        df_full_graph = pd.concat([df_full_graph, df_graph])

    print("Average reward: {}".format(round(sum(l_ep_rew)/len(l_ep_rew)),3))
    env.close()

    model = umap.UMAP(n_components=3)

    # input needs to be (n_samples, n_features)
    # d = tsne_model.fit_transform(activation)
    d_graph = model.fit_transform(graph_embedds.detach().numpy())
    d_node = model.fit_transform(node_embedds.detach().numpy())
    d_linear = model.fit_transform(linear_embedds.detach().numpy())
    df_full_graph['x'] = d_graph[:, 0]
    df_full_graph['y'] = d_graph[:, 1]
    df_full_graph['z'] = d_graph[:, 2]
    df_full_graph['linear_x'] = d_linear[:, 0]
    df_full_graph['linear_y'] = d_linear[:, 1]
    df_full_graph['linear_z'] = d_linear[:, 2]

    df_full['x'] = d_node[:, 0]
    df_full['y'] = d_node[:, 1]
    df_full['z'] = d_node[:, 2]

    app = dash.Dash(__name__)

    fig_full = px.scatter_3d(df_full,
                             x='x',
                             y='y',
                             z='z',
                             color='label',
                             # size='steps',
                             hover_data=
                             {'x': False,
                              'y': False,
                              'z': False,
                              # 'fruit_level': True,
                              # 'agent_level': True,
                              # 'agent_id': True,
                              'steps': True,
                              'episode': True},
                             custom_data = ["images"]
                             )
    fig_full.write_html(f"plots/pp/all_node_embeddings.html")
    # fig_full.update_layout(clickmode='event+select')

    fig_full_graph = px.scatter_3d(df_full_graph,
                                   x='x',
                                   y='y',
                                   z='z',
                                   color=config.hue,
                                   hover_data=
                                   {'x': False,
                                    'y': False,
                                    'z': False,
                                    'q_values': True,
                                    'agent': True,
                                    'steps': True,
                                    'episode': True},
                                   custom_data = ["images"]
                                   )
    fig_full.write_html(f"plots/pp/all_node_embeddings.html")
    # fig_full.update_layout(clickmode='event+select')

    # fig_full_linear = px.scatter_3d(df_full_graph,
    #                                x='linear_x',
    #                                y='linear_y',
    #                                z='linear_z',
    #                                color='q_values',
    #                                # size = 'steps',
    #                                hover_data=
    #                                {'linear_x': False,
    #                                 'linear_y': False,
    #                                 'linear_z': False,
    #                                 'q_values': True,
    #                                 'agent': True,
    #                                 'steps': True,
    #                                 'episode': True},
    #                                custom_data=["images"]
    #                                )
    # fig_full_linear.write_html(f"plots/lbf/all_linear_embeddings.html")


    fig_full_linear = px.scatter_3d(df_full_graph,
                                   x='linear_x',
                                   y='linear_y',
                                   z='linear_z',
                                   color='q_values',
                                   # size = 'steps',
                                   hover_data=
                                   {'linear_x': False,
                                    'linear_y': False,
                                    'linear_z': False,
                                    'q_values': True,
                                    'agent': True,
                                    'steps': True,
                                    'episode': True},
                                   custom_data=["images"]
                                   )
    fig_full_linear.write_html(f"plots/lbf/all_linear_embeddings.html")
    if config.plot_type=='nodes':
        fig = fig_full
    elif config.plot_type=='graph':
        fig = fig_full_graph
    elif config.plot_type=='linear':
        fig = fig_full_linear

    app.layout = html.Div(
        [
            dcc.Graph(
                id="graph_interaction",
                figure=fig,
                style={'display': 'inline-block', 'width': '100vh'}
            ),
            html.Img(id='image', src='',style={'display':'inline-block', 'width': '40vh'})
        ],
    title=f'{config.marc["net_code"]}'
    )

    @app.callback(
        Output('image', 'src'),
        Input('graph_interaction', 'hoverData'))
    def open_url(hoverData):
        if hoverData:
            return hoverData["points"][0]["customdata"][0]
        else:
            raise PreventUpdate

    # @app.callback(
    #     Output('image2', 'src'),
    #     Input('graph_full', 'hoverData'))
    # def open_url(hoverData):
    #     if hoverData:
    #         return hoverData["points"][0]["customdata"][0]
    #     else:
    #         raise PreventUpdate


    return app



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        # default='experiments/MARC/pp/2024-01-24_pp_10x10_2a_1p_2o-v3_b0_std_seed4001/saved_models/ckpt_final.pth.tar',
                        # default='experiments/MARC/pp/2024-03-05_pp_10x10_2a_1p_2o-v3_std_seed4001/saved_models/ckpt_best_avg_r1.149999976158142.pth.tar',
                        # default='experiments/MARC/pp/2024-03-12_pp_5x5_2a_1p_1o-v3_std_seed4001/saved_models/ckpt_best_avg_r*',
                        # default = 'experiments/MARC/pp/2024-03-12_pp_5x5_2a_1p_1o-v3_std_seed4001/saved_models/ckpt_best_avg_r3.626000165939331.pth.tar'
                        help="model_path")
    parser.add_argument("--plot_type", default='nodes')
    parser.add_argument("--eval_n_episodes", default=30, type=int)
    parser.add_argument("--eval_episode_length", default=25, type=int)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render", default=True, action="store_true",
                        help="render")
    parser.add_argument('--hue', default='q_values', help='what hue the plot should have')
    parser.add_argument("--save", default=True, action="store_true",
                        help="save step images")
    parser.add_argument("--benchmark", action="store_false",
                        help="benchmark mode")
    parser.add_argument("--port", default=8050,
                        help="dash port")
    config = parser.parse_args()
    args = vars(config)
    eval_path = Path(config.model_path)
    dir_exp = Path(*eval_path.parts[:-2])
    with open(f"{dir_exp}/config.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    for k,v in params.items():
        args[k] = v

    app = run(config)
    app.run_server(debug=False, port=config.port)

