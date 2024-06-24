import argparse
import torch
import os
import sys
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable
from warnings import filterwarnings  # noqa
from utils.buffer import ReplayBufferMARC, ReplayBufferMAAC, ReplayBufferMPE
from algorithms.attention_sac import AttentionSAC, RelationalSAC
from utils.rel_wrapper2 import AbsoluteVKBWrapper
from utils.env_wrappers import DummyVecEnv
import yaml
from utils.misc import Agent
from utils.plotting import plot_fig
from torch_geometric.data import Data as GeometricData
import wandb

# from multiagent.MPE_env import MPEEnv, GraphMPEEnv
import multiagent as mpe
from multiagent.MPE_env import GraphMPEEnv, MPEEnv
# from multiagent.env_wrappers import GraphDummyVecEnv, GraphSubprocVecEnv
def make_train_env(all_args: argparse.Namespace, n_rollout_threads):
    # def get_env_fn(rank: int):
    #     def init_env():
    if all_args.env_name == "MPE":
        env = mpe.multiagent.MPE_env.MPEEnv(all_args)
    elif all_args.env_name == "GraphMPE":
        env = GraphMPEEnv(all_args)
    else:
        print(f"Can not support the {all_args.env_name} environment")
        raise NotImplementedError
    env.seed(all_args.random_seed * 1000)
    return env

        # return init_env

    # if n_rollout_threads == 1:
    #     if all_args.env_name == "GraphMPE":
    #         return GraphDummyVecEnv([get_env_fn(0)])
    #     return DummyVecEnv([get_env_fn(0)])
    # else:
    #     if all_args.env_name == "GraphMPE":
    #         return GraphSubprocVecEnv(
    #             [get_env_fn(i) for i in range(n_rollout_threads)]
    #         )
    #     return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])



# def make_parallel_env(env_id, n_rollout_threads, seed):
#     def get_env_fn(rank):
#         def init_env():
#             env = make_env(env_id, discrete_action=True)
#             env.seed(seed + rank * 1000)
#             np.random.seed(seed + rank * 1000)
#             return env
#         return init_env
#     if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def to_gd(data: torch.Tensor, unary_t) -> GeometricData:
    """
    takes batch of adjacency geometric data and transforms it to a GeometricData object for torch.geometric

    Parameters
    --------
    data : heterogeneous adjacency matrix (nb_relations, nb_objects, nb_objects)
    """
    # nb_objects = nb_objects
    # x = torch.arange(nb_objects).view(-1, 1) #
    unary_t = torch.tensor(unary_t, dtype=torch.float32)
    data = torch.tensor(data)
    nz = torch.nonzero(data)

    # list of all edges and what relationtype they have
    edge_attr = nz[:, 0]

    # list of node to node indicating an edge
    edge_index = nz[:, 1:].T
    return GeometricData(x=unary_t, edge_index=edge_index, edge_attr=edge_attr)

def run(config):
    torch.set_num_threads(1)
    env_name = config.env_name


    # run_num = 1
    run_dir = config.dir_exp
    # log_dir = config.dir_summary


    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    if config.exp_id == 'std':
        wandb.init(
            project='MARC',
            name=f'{datetime.date.today().day}-{datetime.date.today().month}-{config.alg}-{config.env_id}',
            config=vars(config), )

    env = make_train_env(config, config.n_rollout_threads)
    env.grid_observation = config.grid_observation
    env.agents = [None] * len(env.action_space)
    env.n_agents = len(env.action_space)

    env.n_rel_rules = 5
    obs, agent_id, node_obs, adj = env.reset()

    env.n_attr = np.array(node_obs).shape[2]
    adj = torch.tensor(adj)
    # graph = to_gd(node_obs, adj)
    start_episode = 0
    # env.spatial_tensors = graph
    model = RelationalSAC.init_from_env(env,
                                        spatial_tensors=adj[0,0,:,:],
                                        batch_size = config.batch_size,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       graph_layer = config.marc['graph_layer'],
                                       device=config.device,
                                       reward_scale=config.reward_scale,
                                       dense=config.marc['dense'],
                                       net_code=config.marc['net_code'])

    replay_buffer = ReplayBufferMPE(max_steps=config.marc['buffer_length'],
                                     num_agents=model.n_agents,
                                     obs_dims=[obsp.shape[0] for obsp in env.observation_space],
                                     ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                              for acsp in env.action_space],
                                     dense=config.marc['dense'])

    t = 0
    l_rewards = []
    epymarl_rewards = []
    if config.resume:
        with open(f'{config.resume}/summary/reward_step.txt') as f:
            data = f.readlines()
            steps = int(data[-2].split(',')[0])
        next_step_log = steps + config.step_interval_log
    else:
        steps = 0
        next_step_log = config.step_interval_log
    avg_reward_best = float("-inf")
    path_ckpt_best_avg = '' # TODO could be altered with resume
    for ep_i in range(start_episode, config.n_episodes):
        obs, agent_id, node_obs, adj = env.reset()
        obs = np.expand_dims(obs, axis=0)
        adj = torch.tensor(adj)
        # node_obs = np.array(node_obs)
        graph = [to_gd(adj[agent], node_obs[agent]) for agent in range(env.n_agents)]
        model.prep_rollouts(device='cpu')
        episode_reward_total = 0
        is_best_avg = False

        for et_i in range(1, config.episode_length + 1):
            torch_obs = [Variable(torch.Tensor((obs[:,i])),
                                      requires_grad=False)
                             for i in range(model.n_agents)]

            # get actions as torch Variables
            try:
                torch_agent_actions = model.step(torch_obs, explore=True)
            except Exception as e:
                print(e)
                continue
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            # rearrange actions to be per environment
            # if config.alg == 'MARC':
            #     actions = [np.argmax(ac) for ac in agent_actions]
            # else:
            # print('agent_actions', agent_actions)
            actions = [np.argmax(ac[0]) for ac in agent_actions]
            next_obs, agent_id, node_obs, adj, rewards, dones, infos = env.step(actions)
            next_obs = np.expand_dims(next_obs, axis=0)
            rewards, dones = np.array(rewards), np.array(dones)
            adj = torch.tensor(adj)
            next_graph = [to_gd(adj[agent], node_obs[agent]) for agent in range(env.n_agents)]

            # obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n
            # rewards, dones = np.array(rewards), np.array(dones)

            # next_obs = tuple([next_obs[:,i][0] for i in range(model.n_agents)])
            # next_obs = np.vstack(next_obs)
            # next_obs = np.expand_dims(next_obs, axis=0)

            replay_buffer.push(obs, graph, agent_actions, rewards, next_obs, next_graph, dones)


            episode_reward_total += rewards.sum()
            obs = next_obs
            graph = next_graph
            t += 1
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) ==0):
                model.prep_training(device=config.device)
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  device=config.device, norm_rews=config.norm_rews)
                    model.update_critic(sample)
                    model.update_policies(sample)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')

            if dones.all() == True:
                print('done with time step', et_i)
                break
        steps += et_i
        try:
            wandb.log({
                "rew": episode_reward_total,
                "obst_collisions": sum([infos[0,i]['Num_obst_collisions'] for i in range(infos.shape[1])]),
                "agent_collisions": sum([infos[0,i]['Num_agent_collisions'] for i in range(infos.shape[1])]),
                "success": np.mean([1 if infos[0,i]['individual_reward']== 5 else 0 for i in range(infos.shape[1])]),
                'time_to_goal': np.mean([infos[0,i]['Time_req_to_goal'] for i in range(infos.shape[1])]),
                "steps": et_i})

        except:
            pass
        print("%s - %s - Episodes %i (%is) of %i - Reward %.2f" % (config.env_id, config.random_seed, ep_i + 1, steps,
                                        config.n_episodes,  episode_reward_total))
        l_rewards.append(episode_reward_total)
        epymarl_rewards.append(episode_reward_total)
        # check if it in average was the best model so far
        th_l_rewards = torch.FloatTensor(np.asarray(l_rewards))


        if len(th_l_rewards) >= 100:
            avg_rewards = th_l_rewards.unfold(0, 100, 1).mean(1).view(-1)
            avg_rewards = torch.cat((torch.zeros(99), avg_rewards))
            avg_reward = avg_rewards[-1]
            if avg_reward > avg_reward_best:
                avg_reward_best = avg_reward
                is_best_avg = True

            os.makedirs('{}/summary/'.format(run_dir), exist_ok=True)

            if steps >= next_step_log:
                try:
                    wandb.log({
                               'epymarl_return_mean': np.mean(epymarl_rewards),
                               'steps': steps})
                except:
                    pass
                with open("{}/summary/reward_step.txt".format(run_dir), "a") as f:
                    f.write("{},{} \n".format(steps, avg_reward))
                with open("{}/summary/reward_epymarl.txt".format(run_dir), "a") as f:
                    f.write("{},{} \n".format(steps, np.mean(epymarl_rewards)))
                epymarl_rewards.clear()
                next_step_log += config.step_interval_log

            if ep_i % config.save_interval_log == 0:
                with open('{}/summary/reward_total.txt'.format(run_dir), 'w') as fp:
                    for el in l_rewards:
                        fp.write("{}\n".format(round(el, 2)))
                if ep_i % 5000 == 0:
                    path_ckpt_eps_tmp = os.path.join(config.dir_saved_models, 'ckpt_ep{}_{}.pth.tar'.format(ep_i, avg_reward))
                    model.save(filename=path_ckpt_eps_tmp, episode=ep_i)
            if is_best_avg:
                path_ckpt_best_avg_tmp = os.path.join(config.dir_saved_models,
                                                      'ckpt_best_avg_r{}.pth.tar'.format(avg_reward_best))
                model.save(filename=path_ckpt_best_avg_tmp, episode=ep_i)
                # torch.save(ckpt, path_ckpt_best_avg_tmp)
                if os.path.exists(path_ckpt_best_avg):
                    os.remove(path_ckpt_best_avg)
                path_ckpt_best_avg = path_ckpt_best_avg_tmp

            if ep_i % config.test_interval ==0:
                continue
                # TODO this needs to be adjusted
                model.prep_rollouts(device='cpu')
                l_ep_rew = []
                for eval_ep_i in range(config.test_n_episodes):
                    ep_rew = 0

                    obs = env.reset()
                    for t_i in range(config.episode_length):

                        # rearrange observations to be per agent, and convert to torch Variable

                        if config.alg == 'MARC':
                            obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
                            torch_obs = [Variable(torch.Tensor(obs[i]),
                                                  requires_grad=False)
                                         for i in range(model.n_agents)]
                            # get actions as torch Variables
                            torch_actions = model.step(torch_obs, explore=False)
                            # convert actions to numpy arrays
                            actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
                        else:
                            if env.grid_observation:
                                obs = tuple([obs[:, i][0]['image'].flatten() for i in range(model.n_agents)])
                                obs = np.vstack(obs)
                                obs = np.expand_dims(obs, axis=0)
                            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                                  requires_grad=False)
                                         for i in range(model.n_agents)]
                            torch_agent_actions = model.step(torch_obs, explore=False)
                            # convert actions to numpy arrays
                            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
                            # rearrange actions to be per environment
                            actions = [[np.argmax(ac[0]) for ac in agent_actions]]

                        obs, rewards, dones, infos = env.step(actions)
                        rewards, dones = np.array(rewards), np.array(dones)
                        ep_rew += rewards.sum()
                        if dones.all():
                            break

                    l_ep_rew.append(ep_rew)
                    print("Episode %i of %i - %i" % (eval_ep_i + 1, config.test_n_episodes, ep_rew))

                print('test: ',np.mean(l_ep_rew), 'train: ', np.mean(l_rewards[-10:]))
                avg_eval_rew = sum(l_ep_rew) / config.test_n_episodes
                print("Average eval reward: {}".format(avg_eval_rew))
                with open('{}/summary/eval_reward.txt'.format(run_dir), 'a') as file:
                    file.write("{}\n".format(round(avg_eval_rew, 2)))
                try:
                    wandb.log({'avg_eval_return': avg_eval_rew, 'eval_at_step': ep_i})
                except:
                    pass

    # plot_fig(l_rewards, 'reward_total', config.dir_summary, show=True)
    path_ckpt_final = os.path.join(config.dir_saved_models,
                                          'ckpt_final.pth.tar')
    model.save(filename=path_ckpt_final, episode=ep_i)
    # model.save('{}/model.pt'.format(config.dir_saved_models))
    env.close()
    # logger.export_scalars_to_json('{}/summary.json'.format(run_dir))
    # logger.close()
    wandb.finish()

# def make_parallel_MAAC_env(args, seed):
#     def get_env_fn(rank):
#         def init_env():
#             env = make_env(args)
#             env.agents = [Agent() for _ in range(args.player)]
#             # env.grid_observation = args.grid_observation
#             # env.seed(args.random_seed + rank * 1000)
#             np.random.seed(args.random_seed + rank * 1000)
#             return env
#         return init_env
#     # if config.n_rollout_threads == 1:
#     return DummyVecEnv([get_env_fn(0)])
#     # else:
#     #     return SubprocVecEnv([get_env_fn(i) for i in range(config.n_rollout_threads)])
#


if __name__ == '__main__':
    import shutil
    import datetime
    import time
    import json
    import glob as glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str)
    parser.add_argument("--resume_episodes", type=int)
    parser.add_argument("--random_seed", default=4001, type=int)
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=128, type=int,
                        help="Batch size for training")
    parser.add_argument("--test_interval", default=1000, type=int)
    parser.add_argument("--save_interval_log", default=100, type=int)
    parser.add_argument('--step_interval_log', default=10000, type=int)

    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)

    parser.add_argument("--pi_lr", default=0.001, type=float) # learning rate policy
    parser.add_argument("--q_lr", default=0.001, type=float) # leanring rate critic
    parser.add_argument("--tau", default=0.001, type=float) # soft update rate
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float) # temperature parameter alpha = 1/reward_scale = 0.01 in this case
    # parser.add_argument("--device",default='cuda:0', type=str)
    parser.add_argument('--dir_base', default='./experiments',
                        help='path of the experiment directory')
    config = parser.parse_args()
    if torch.cuda.is_available():
        print('cuda is available')
        config.use_gpu = True
    else:
        config.use_gpu = False
    args = vars(config)
    if not args['resume']:
        with open('config.yaml', "r") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
    else:
        with open(f"{args['resume']}/config.yaml", "r") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            params['resume'] = args['resume']
            params['resume_episodes'] = args['resume_episodes']

    for k, v in params.items():
        args[k] = v


    # config.seed = 4001

    print(f'using {config.device}')
    # deletes arguments that are not used in this experiment
    if not args['resume']:

        config.num_agents = config.player
        args['env_id'] = f"MAPE-{args['num_agents']}p{args['other']}"


        if params['exp_id'] == 'try':
            args['env_id'] = 'TEST'
            args['n_episodes']= 2000
            args['episode_length']= 25
            args['test_n_episodes']= 10
            args['maac']['buffer_length'] = 1100
            args['marc']['buffer_length'] = 1100
            args['test_interval'] = 100
            args['step_interval_log'] = 200

        if args['alg'] == 'MARC':
            del args['maac']
        else:
            del args['marc']

    dir_collected_data = './experiments/multipleseeds_data_{}_{}_{}'.format(args['alg'], args['env_id'],
                                                                                 args['exp_id'])

    if os.path.exists(dir_collected_data):
        toDelete = 'yes'
        if toDelete.lower() == 'yes':
            shutil.rmtree(dir_collected_data)
            print("Directory removed")
            os.makedirs(dir_collected_data)
    else:
        os.makedirs(dir_collected_data)

    list_exp_dir = []
    if not args['resume']:

        dir_exp_name = '{}_{}_{}_seed{}'.format(str([datetime.date.today()][0]),
                                                args['env_id'],
                                                args['exp_id'],
                                                args['random_seed'])
        args['dir_exp'] = '{}/{}/{}/{}'.format(args['dir_base'], args['alg'],args['env_name'], dir_exp_name)
        args['dir_summary'] = '{}/summary'.format(args['dir_exp'])
        args['dir_saved_models'] = '{}/saved_models'.format(args['dir_exp'])
        args['dir_monitor'] = '{}/monitor'.format(args['dir_exp'])

        # creating folders:
        directory = args['dir_exp']
        if os.path.exists(directory):
            # toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
            #     format(directory))
            toDelete = 'yes'
            if toDelete.lower() == 'yes':
                shutil.rmtree(directory)
                print("Directory removed")
            if toDelete.lower() == 'no':
                print("It was not possible to continue, an experiment \
                        folder is required.Terminating here.")
                import sys

                sys.exit()

        if os.path.exists(directory) == False:
            os.makedirs(directory)
            os.makedirs(args['dir_summary'])
            os.makedirs(args['dir_saved_models'])
            os.makedirs(args['dir_monitor'])
            # with open(os.path.expanduser('{}/arguments.txt'.format(args['dir_exp'])), 'w+') as file:
            #     file.write(json.dumps(args, indent=4, sort_keys=True))
            with open(os.path.expanduser('{}/config.yaml'.format(args['dir_exp'])), 'w+') as file:
                documents = yaml.dump(args, file)


        # train
        list_exp_dir.append(args['dir_exp'])
        print('before run ok')
        if os.path.exists('{}/collected_data_seed_{}.json'.format(dir_collected_data, args['random_seed'])) == False:
            st = time.time()
            run(config)

            # test
            # model_path = '{}/'.format(args['dir_exp'])
            model_path = glob.glob('{}/saved_models/ckpt_best_*'.format(args['dir_exp']))[0]
            t_min = (time.time() - st)/60
            print(f'{args["n_episodes"]} episode on gpu:{args["use_gpu"]} with max {args["episode_length"]} steps ran in {t_min:.2f} minutes - {t_min*60/args["n_episodes"]:.2f} sec/episode')

            cmd_test = 'python evaluate_discrete.py {}'.format(model_path)
            print(cmd_test)

            os.system(cmd_test)

            shutil.copyfile('{}/summary/reward_total.txt'.format(args['dir_exp']), # TODO check if this works - changed  list_exp_dir[-1]
                            '{}/reward_training_seed{}.txt'.format(dir_collected_data, args['random_seed'])
                            )
    else:
        if os.path.exists('{}/collected_data_seed_{}.json'.format(dir_collected_data, args['random_seed'])) == False:
            st = time.time()
            run(config)

            # test
            # model_path = '{}/'.format(args['dir_exp'])
            model_path = glob.glob('{}/saved_models/ckpt_best_*'.format(args['dir_exp']))[0]
            t_min = (time.time() - st) / 60
            print(
                f'{args["n_episodes"]} episode on gpu:{args["use_gpu"]} with max {args["episode_length"]} steps ran in {t_min:.2f} minutes - {t_min * 60 / args["n_episodes"]:.2f} sec/episode')

            if config.use_gpu:
                cmd_test = 'python evaluate_discrete.py {} '.format(model_path)
            else:
                cmd_test = 'python evaluate_discrete.py {} --render'.format(model_path)
            print(cmd_test)

            os.system(cmd_test)

