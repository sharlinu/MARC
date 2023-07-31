import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from warnings import filterwarnings  # noqa
filterwarnings(action='ignore',
                        category=DeprecationWarning,
                        module='tensorboardX')
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer, ReplayBuffer2
from algorithms.attention_sac import RelationalSAC
from utils.rel_wrapper2 import AbsoluteVKBWrapper
import yaml
from utils.plotting import plot_fig
from lbforaging.foraging import ForagingEnv

def run(config):
    torch.set_num_threads(1)

    run_num = 1
    run_dir = config.dir_exp
    log_dir = config.dir_summary

    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    env = ForagingEnv(
        players=config.player,
        max_player_level=config.max_player_level,
        field_size=(config.field, config.field),
        max_food=config.max_food,
        grid_observation=config.grid_observation,
        sight=config.field,
        max_episode_steps=config.episode_length,
        force_coop=config.force_coop,
        keep_food=config.keep_food,
        simple=config.simple,
    )
    env.seed(config.random_seed)
    np.random.seed(config.random_seed)
    env = AbsoluteVKBWrapper(env, config.dense)
    env.agents = [None] * len(env.action_space)
    # nullary_dim = env.obs_shape[0]
    unary_dim = env.obs_shape['unary']
    # binary_dim = env.obs_shape[2]
    env.reset()
    spatial_tensors = env.spatial_tensors
    model = RelationalSAC.init_from_env(env,
                                       spatial_tensors=spatial_tensors,
                                       batch_size = config.batch_size,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       reward_scale=config.reward_scale)

    replay_buffer = ReplayBuffer2(max_steps=config.buffer_length,
                                 num_agents=model.n_agents,
                                 obs_dims=[np.prod(obsp['image'].shape) for obsp in env.observation_space],
                                 # nullary_dims=[nullary_dim for _ in range(model.nagents)],
                                 unary_dims=[unary_dim for _ in range(model.n_agents)],
                                 # binary_dims=[binary_dim for _ in range(model.nagents)],
                                 ac_dims= [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                 for acsp in env.action_space],
                                 dense = config.dense)
    t = 0
    l_rewards = []
    reward_best = float("-inf")
    avg_reward = float("-inf")
    avg_reward_best = float("-inf")
    path_ckpt_best_avg = ''
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        # print('player level',[p.level for p in env.players])
        # print('field', env.field)
        #print('initial', obs['image'][0],obs['image'][1])
        model.prep_rollouts(device='cpu')
        episode_reward_total = 0
        is_best_avg          = False

        for et_i in range(1, config.episode_length + 1):
            # rearrange observations to be per agent, and convert to torch Variable
            agent_obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
            #agent_obs = obs['image'].flatten()
            #agent_obs = np.expand_dims(agent_obs, axis=0)
            torch_obs = [Variable(torch.Tensor(agent_obs[i]),
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
            # rearrange actions to be per environment
            actions = [np.argmax(ac) for ac in agent_actions]
            next_obs, rewards, dones, infos = env.step(actions)
            # print('player level', [p.level for p in env.players])
            # print('field', env.field)
            # print('reward', rewards)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            episode_reward_total += rewards.sum()

            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                    (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            if dones.all() == True:
                print('done with time step', et_i)
                break
            # TODO change back

        if ('box' or 'collect') in config.env_id:
            ep_rews = replay_buffer.get_average_rewards(et_i)
        else:
            ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
             logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        print("%s - %s - Episodes %i of %i - Reward %.1f" % (config.env_id, config.random_seed, ep_i + 1,
                                        config.n_episodes,  episode_reward_total))
        l_rewards.append(episode_reward_total)
        #print('terminal space', obs['image'])


        # check if it in average was the best model so far
        th_l_rewards = torch.FloatTensor(np.asarray(l_rewards))


        if len(th_l_rewards) >= 100:
            avg_rewards = th_l_rewards.unfold(0, 100, 1).mean(1).view(-1)
            avg_rewards = torch.cat((torch.zeros(99), avg_rewards))
            avg_reward = avg_rewards[-1]
            if avg_reward > avg_reward_best:
                avg_reward_best = avg_reward
                is_best_avg = True

            if ep_i % config.save_interval_log == 0:
                os.makedirs('{}/summary/'.format(run_dir), exist_ok=True)
                with open('{}/summary/reward_total.txt'.format(run_dir), 'w') as fp:
                    for el in l_rewards:
                        fp.write("{}\n".format(round(el, 2)))
                plot_fig(l_rewards, 'reward_total', config.dir_summary)

            if is_best_avg:
                path_ckpt_best_avg_tmp = os.path.join(config.dir_saved_models,
                                                      'ckpt_best_avg_r{}.pth.tar'.format(avg_reward_best))
                model.save(path_ckpt_best_avg_tmp)
                # torch.save(ckpt, path_ckpt_best_avg_tmp)
                if os.path.exists(path_ckpt_best_avg):
                    os.remove(path_ckpt_best_avg)
                path_ckpt_best_avg = path_ckpt_best_avg_tmp

            if ep_i % config.save_interval ==0:
                l_ep_rew = []
                for eval_ep_i in range(config.test_n_episodes):
                    print("Episode %i of %i" % (eval_ep_i + 1, config.test_n_episodes))

                    ep_rew = 0

                    # from utils.rel_wrapper2 import AbsoluteVKBWrapper
                    # env = AbsoluteVKBWrapper(env, config.dense)

                    obs = env.reset()

                    for t_i in range(config.test_episode_length):

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

                        ep_rew += sum(rewards)

                        if all(dones):
                            break

                    l_ep_rew.append(ep_rew)
                    # print("Reward: {}".format(ep_rew))


                avg_eval_rew = sum(l_ep_rew) / config.test_n_episodes
                print("Average eval reward: {}".format(avg_eval_rew))
                with open('{}/summary/eval_reward.txt'.format(run_dir), 'a') as file:
                    file.write("{}\n".format(round(avg_eval_rew, 2)))

    plot_fig(l_rewards, 'reward_total', config.dir_summary, show=True)
    path_ckpt_final = os.path.join(config.dir_saved_models,
                                          'ckpt_final.pth.tar')
    model.save(path_ckpt_final)
    # model.save('{}/model.pt'.format(config.dir_saved_models))
    env.close()
    logger.export_scalars_to_json('{}/summary.json'.format(run_dir))
    logger.close()

if __name__ == '__main__':
    import shutil
    import datetime
    import time
    import json
    import glob as glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=1, type=int)
    # parser.add_argument("model_name",
    #                     help="Name of directory to store " +
    #                          "model/training contents")
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
    parser.add_argument("--save_interval", default=200, type=int)
    parser.add_argument("--save_interval_log", default=100, type=int)

    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument('--dir_base', default='./experiments',
                        help='path of the experiment directory')
    config = parser.parse_args()

    if torch.cuda.is_available():
        print('cuda is available')
        config.use_gpu = True
    else:
        config.use_gpu = False

    args = vars(config)
    with open("config.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    for k, v in params.items():
        args[k] = v
    args['env_id'] = f"{args['env']}_{args['field']}x{args['field']}_{args['player']}p_{args['max_food']}f{'_coop' if args['force_coop'] else ''}{args['other']}"

    dir_collected_data = './experiments/MAAC_multipleseeds_data_{}_{}_{}'.format(args['agent_alg'], args['env_id'],
                                                                                 args['exp_id'])

    if os.path.exists(dir_collected_data):
        #toDelete = input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ". \
        #                 format(dir_collected_data))
        toDelete = 'yes'
        if toDelete.lower() == 'yes':
            shutil.rmtree(dir_collected_data)
            print("Directory removed")
            os.makedirs(dir_collected_data)
    else:
        os.makedirs(dir_collected_data)

    list_exp_dir = []
    for seed in args['seeds']:
        args['random_seed'] = seed

        dir_exp_name = '{}_{}_{}_seed{}'.format(str([datetime.date.today()][0]),
                                                args['env_id'],
                                                args['exp_id'],
                                                args['random_seed'])
        args['dir_exp'] = '{}/{}'.format(args['dir_base'], dir_exp_name)
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
                        folder is required.Terminiting here.")
                import sys

                sys.exit()
        if os.path.exists(directory) == False:
            os.makedirs(directory)
            os.makedirs(args['dir_summary'])
            os.makedirs(args['dir_saved_models'])
            os.makedirs(args['dir_monitor'])
            with open(os.path.expanduser('{}/arguments.txt'.format(args['dir_exp'])), 'w+') as file:
                file.write(json.dumps(args, indent=4, sort_keys=True))
            with open(os.path.expanduser('{}/config.yaml'.format(args['dir_exp'])), 'w+') as file:
                documents = yaml.dump(args, file)


        # train
        list_exp_dir.append(args['dir_exp'])

        if os.path.exists('{}/collected_data_seed_{}.json'.format(dir_collected_data, seed)) == False:
            st = time.time()
            run(config)

            # test
            # model_path = '{}/'.format(args['dir_exp'])
            model_path = glob.glob('{}/saved_models/ckpt_best_*'.format(args['dir_exp']))[0]
            t_min = (time.time() - st)/60
            print(f'{args["n_episodes"]} episode on gpu:{args["use_gpu"]} with max {args["episode_length"]} steps ran in {t_min:.2f} minutes - {t_min*60/args["n_episodes"]:.2f} sec/episode')
            cmd_test = 'python evaluate_discrete.py {} --no_render'.format(model_path)
            print(cmd_test)
            os.system(cmd_test)

            # save files to dir collected data
            #            shutil.copyfile('{}/evaluate/collected_data.json'.format(list_exp_dir[-1]),
            #                    '{}/collected_data_seed_{}.json'.format(dir_collected_data,seed)
            #                    )

            # save files to dir collected data
            shutil.copyfile('{}/summary/reward_total.txt'.format(list_exp_dir[-1]),
                            '{}/reward_training_seed{}.txt'.format(dir_collected_data, seed)
                            )
