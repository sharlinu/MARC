import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC, RelationalSAC
from utils.rel_wrappers import AbsoluteVKBWrapper
import gym
def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            #env = gym.make("Foraging-8x8-2p-1f-v0")
            from r_maac.box import BoxWorldEnv
            env = BoxWorldEnv(
                players=2,
                field_size=(4,4),
                num_colours=2,
                goal_length=2,
                sight=4,
                max_episode_steps=500,
                grid_observation=True,
                simple=True,
                relational=False,
                #deterministic=True,
            )
            env = AbsoluteVKBWrapper(env,num_colours=env.num_colours)
            env.agents = [None] * len(env.action_space)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config, configvar):
    torch.set_num_threads(1)

    # model_dir = Path('./models') / config.env_id / config.model_name
    # if not model_dir.exists():
    #     run_num = 1
    # else:
    #     exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
    #                      model_dir.iterdir() if
    #                      str(folder.name).startswith('run')]
    #     if len(exst_run_nums) == 0:
    #         run_num = 1
    #     else:
    #         run_num = max(exst_run_nums) + 1

    # curr_run = 'run%i' % run_num
    # run_dir = model_dir / curr_run
    # log_dir = run_dir / 'logs'
    # os.makedirs(str(log_dir))

    run_num = 1
    run_dir = configvar['dir_exp']
    log_dir = configvar['dir_summary']
    config.env_id = configvar['env_id']

    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(configvar['random_seed'])
    np.random.seed(configvar['random_seed'])
    #env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    from r_maac.box import BoxWorldEnv
    env = BoxWorldEnv(
        players=2,
        field_size=(4, 4),
        num_colours=2,
        goal_length=2,
        sight=4,
        max_episode_steps=500,
        grid_observation=True,
        simple=True,
        relational=False,
        # deterministic=True,
    )
    env = AbsoluteVKBWrapper(env, num_colours=env.num_colours)
    env.agents = [None] * len(env.action_space)
    nullary_dim = env.obs_shape[0]
    unary_dim = env.obs_shape[1]
    binary_dim = env.obs_shape[2]
    env.seed(seed)
    np.random.seed(seed)

    # model = AttentionSAC.init_from_env(env,
    #                                    tau=config.tau,
    #                                    pi_lr=config.pi_lr,
    #                                    q_lr=config.q_lr,
    #                                    gamma=config.gamma,
    #                                    pol_hidden_dim=config.pol_hidden_dim,
    #                                    critic_hidden_dim=config.critic_hidden_dim,
    #                                    attend_heads=config.attend_heads,
    #                                    reward_scale=config.reward_scale)
    model = RelationalSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       reward_scale=config.reward_scale)

    replay_buffer = ReplayBuffer(max_steps=config.buffer_length,
                                 num_agents=model.nagents,
                                 obs_dims=[np.prod(obsp['image'].shape) for obsp in env.observation_space],
                                 nullary_dims=[nullary_dim for _ in range(model.nagents)],
                                 unary_dims=[unary_dim for _ in range(model.nagents)],
                                 binary_dims=[binary_dim for _ in range(model.nagents)],
                                 ac_dims= [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    # TODO change replay buffer
    t = 0
    l_rewards = []
    reward_best      = float("-inf")
    avg_reward       = float("-inf")
    avg_reward_best  = float("-inf")
    path_ckpt_best_avg = ''
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("%s - %s - Episodes %i-%i of %i" % (env_id, configvar['random_seed'], ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        model.prep_rollouts(device='cpu')
        episode_reward_total = 0
        is_best_avg          = False

        for et_i in range(1, config.episode_length+1):
            # rearrange observations to be per agent, and convert to torch Variable
            agent_obs = obs['image'].flatten()
            agent_obs = np.expand_dims(agent_obs, axis=0)
            torch_obs = [Variable(torch.Tensor(agent_obs),
                                  requires_grad=False)
                         for _ in range(model.nagents)]
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
            #replay_buffer.push(obs, agent_actions, rewards, next_obs, dones) # TODO change replay buffer
            episode_reward_total += rewards.sum()

            # TODO add break for dones / episodic tasks
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
#        ep_rews = replay_buffer.get_average_rewards(
#            config.episode_length * config.n_rollout_threads) # TODO change replay buffer
        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        l_rewards.append(episode_reward_total)


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

            if is_best_avg:
                path_ckpt_best_avg_tmp = os.path.join(configvar['dir_saved_models'], 
                                        'ckpt_best_avg_r{}.pth.tar'.format(avg_reward_best))
                model.save(path_ckpt_best_avg_tmp)
                # torch.save(ckpt, path_ckpt_best_avg_tmp)          
                if os.path.exists(path_ckpt_best_avg):
                    os.remove(path_ckpt_best_avg)
                path_ckpt_best_avg = path_ckpt_best_avg_tmp



    model.save('{}/model.pt'.format(run_dir))
    env.close()
    logger.export_scalars_to_json('{}/summary.json'.format(run_dir))
    logger.close()


if __name__ == '__main__':
    import shutil 
    import datetime
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
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=10000, type=int)
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
    parser.set_defaults(use_gpu=False)
    config = parser.parse_args()
    args   = vars(config)

    env_id = 'Boxworld'
    agent_alg = 'MAAC'

    exp_id = 'std'
    # exp_id = 'try'
    if exp_id == 'try':
        seeds = [1,2001]
        train_n_episodes = 101
        train_episode_length = 25
        test_n_episodes = 5
        test_episode_length = 25
    else:
        seeds = [2001,4001] #[1,2001,4001,6001,8001]
        train_n_episodes = 4000
        train_episode_length = 25
        test_n_episodes = 2
        test_episode_length = 25
    
    dir_collected_data = './experiments/MAAC_multipleseeds_data_{}_{}_{}'.format(agent_alg,env_id,exp_id)
    if os.path.exists(dir_collected_data):
        toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
            format(dir_collected_data))
        if toDelete.lower() == 'yes':
            shutil.rmtree(dir_collected_data)
            print("Directory removed")
            os.makedirs(dir_collected_data)
    else:
        os.makedirs(dir_collected_data)

    list_exp_dir = []
    for seed in seeds:
        args['env_id']         = env_id
        args['random_seed']    = seed
        args['n_episodes']     = train_n_episodes
        args['episode_length'] = train_episode_length
        args['exp_id']         = exp_id
        args['modelname']      = 'MAAC'
        args['modelname']      = 'MAAC'

        dir_exp_name = '{}_{}_{}_{}_{}_seed{}'.format(str([datetime.date.today()][0]),
                                    args['env_id'],
                                    args['modelname'],
                                    agent_alg,
                                    args['exp_id'],
                                    args['random_seed'])
        args['dir_exp'] = '{}/{}'.format(args['dir_base'],dir_exp_name)
        args['dir_summary'] = '{}/summary'.format(args['dir_exp'])
        args['dir_saved_models'] = '{}/saved_models'.format(args['dir_exp'])
        args['dir_monitor'] = '{}/monitor'.format(args['dir_exp'])

        # creating folders:
        directory = args['dir_exp']
        if os.path.exists(directory):
            # toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
            #     format(directory))
            toDelete = 'no'
            if toDelete.lower() == 'yes':
                shutil.rmtree(directory)
                print("Directory removed")
            if toDelete == 'No':
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

        # train
        list_exp_dir.append(args['dir_exp'])

        if  os.path.exists('{}/collected_data_seed_{}.json'.format(dir_collected_data,seed)) == False:
            run(config, args)
            
            # test
            # model_path = '{}/'.format(args['dir_exp'])
            #model_path = glob.glob('{}/saved_models/ckpt_best_*'.format(args['dir_exp']))[0]
            #cmd_test = 'python evaluate.py {} {} --n_episodes {} --episode_length {} --no_render'.format(env_id, model_path, test_n_episodes, test_episode_length)
            #print(cmd_test)
            #os.system(cmd_test)

            # save files to dir collected data
            shutil.copyfile('{}/evaluate/collected_data.json'.format(list_exp_dir[-1]),
                    '{}/collected_data_seed_{}.json'.format(dir_collected_data,seed)
                    )

            # save files to dir collected data
            shutil.copyfile('{}/summary/reward_total.txt'.format(list_exp_dir[-1]),
                    '{}/reward_training_seed{}.txt'.format(dir_collected_data,seed)
                    )
