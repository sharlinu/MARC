import argparse
import torch
import os
import numpy as np
from gym.spaces import Box
from torch.autograd import Variable
from warnings import filterwarnings  # noqa
# filterwarnings(action='ignore',
#                         category=DeprecationWarning,
#                         module='gym') # TODO update
from utils.buffer import ReplayBufferMARC, ReplayBufferMAAC
from algorithms.attention_sac import AttentionSAC, RelationalSAC
from utils.rel_wrapper2 import AbsoluteVKBWrapper
from utils.env_wrappers import DummyVecEnv
import yaml
from utils.misc import Agent
from utils.plotting import plot_fig
import wandb



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=20000, type=int)
    parser.add_argument("--test_n_episodes", default=100, type=int)
    # parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--test_interval", default=1000, type=int)
    parser.add_argument("--save_interval_log", default=100, type=int)
    parser.add_argument('--step_interval_log', default=1000, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float) # temperature parameter alpha = 1/reward_scale = 0.01 in this case
    parser.add_argument("--tau", default=0.001,type=float)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--alg", default='MAAC', type=str)

    default_config = parser.parse_args()
    if torch.cuda.is_available():
        print('cuda is available')
        default_config.use_gpu = True
    else:
        default_config.use_gpu = False
    args = vars(default_config)
    # with open('config.yaml', "r") as file:
    #     params = yaml.load(file, Loader=yaml.FullLoader)
    # for k, v in params.items():
    #     args[k] = v

    args['alg'] = 'MAAC'
    args['env_id'] = "lbf_15x15_3_5f_keep_food"
    args['random_seed'] = 4001
    args['n_episodes'] = 100000
    args['episode_length'] = 50
    args['player'] = 3
    args['max_food'] = 5


    torch.set_num_threads(1)

    wandb.init(config=None)
    hyper_config = wandb.config

    env_name = 'lbf'
    start_episode = 1


    torch.manual_seed(default_config.random_seed)
    np.random.seed(default_config.random_seed)
    if default_config.alg == 'MARC':
        env = make_env(default_config)
        env.grid_observation = default_config.grid_observation
        attr_mapping = getattr(default_config, env_name)['attr_mapping']
        env = AbsoluteVKBWrapper(env=env,
                                 attr_mapping=attr_mapping,
                                 dense=default_config.marc['dense'],
                                 background_id=default_config.marc['background_id'],
                                 abs_id=default_config.marc['abs_id']
                                 )
        env.agents = [None] * len(env.action_space)
        unary_dim = env.obs_shape['unary']
        env.reset()
        model = RelationalSAC.init_from_env(env,
                                            spatial_tensors=env.spatial_tensors,
                                            batch_size = hyper_config.batch_size,
                                           tau=hyper_config.tau,
                                           pi_lr=hyper_config.pi_lr,
                                           q_lr=hyper_config.q_lr,
                                           gamma=default_config.gamma,
                                           pol_hidden_dim=hyper_config.pol_hidden_dim,
                                           critic_hidden_dim=hyper_config.critic_hidden_dim,
                                           reward_scale=default_config.reward_scale)
        replay_buffer = ReplayBufferMARC(max_steps=default_config.marc['buffer_length'],
                                         num_agents=model.n_agents,
                                         obs_dims=[np.prod(obsp['image'].shape) for obsp in env.observation_space],
                                         # nullary_dims=[nullary_dim for _ in range(model.nagents)],
                                         unary_dims=[unary_dim for _ in range(model.n_agents)],
                                         # binary_dims=[binary_dim for _ in range(model.nagents)],
                                         ac_dims=[acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                                  for acsp in env.action_space],
                                         dense=default_config.marc['dense'])

    elif default_config.alg == 'MAAC':
        env = make_parallel_MAAC_env(default_config)
        env.grid_observation = False
        env.reset()
        model = AttentionSAC.init_from_env(env,
                                           tau=hyper_config.tau,
                                           pi_lr=hyper_config.pi_lr,
                                           q_lr=hyper_config.q_lr,
                                           gamma=default_config.gamma,
                                           pol_hidden_dim=hyper_config.pol_hidden_dim,
                                           critic_hidden_dim=hyper_config.critic_hidden_dim,
                                           attend_heads=hyper_config.attend_heads,
                                           reward_scale=default_config.reward_scale)
        replay_buffer = ReplayBufferMAAC(default_config.buffer_length, model.nagents,
                                     [np.prod(obsp['image'].shape) if env.grid_observation else obsp.shape[0]
                                      for obsp in env.observation_space],
                                     [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                      for acsp in env.action_space])
    else:
        raise ValueError(f'Cannot identify algorithm {default_config.alg}')

    t = 0
    l_rewards = []
    epymarl_rewards = []
    steps = 0
    avg_reward_best = float("-inf")
    path_ckpt_best_avg = ''
    for ep_i in range(start_episode, default_config.n_episodes+1):
        obs = env.reset()
        if env.grid_observation and default_config.alg =='MAAC':
            obs = tuple([obs[:, i][0]['image'].flatten() for i in range(model.nagents)])
            obs = np.vstack(obs)
            obs = np.expand_dims(obs, axis=0)

        model.prep_rollouts(device='cpu')
        episode_reward_total = 0
        is_best_avg = False

        for et_i in range(1, default_config.episode_length + 1):
            if default_config.alg == 'MARC':
                # rearrange observations to be per agent, and convert to torch Variable
                agent_obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
                torch_obs = [Variable(torch.Tensor(agent_obs[i]),
                                      requires_grad=False)
                             for i in range(model.n_agents)]
            else:
                torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                      requires_grad=False)
                             for i in range(model.nagents)]
            # get actions as torch Variables
            try:
                torch_agent_actions = model.step(torch_obs, explore=True)
            except Exception as e:
                print(e)
                continue
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environmentsweep.py
            if default_config.alg == 'MARC':
                actions = [np.argmax(ac) for ac in agent_actions]
            else:
                actions = [[np.argmax(ac[0]) for ac in agent_actions]]
            next_obs, rewards, dones, infos = env.step(actions)
            rewards, dones = np.array(rewards), np.array(dones)

            if default_config.alg == 'MAAC' and env.grid_observation:
                next_obs = tuple([next_obs[:,i][0]['image'].flatten() for i in range(model.nagents)])
                next_obs = np.vstack(next_obs)
                next_obs = np.expand_dims(next_obs, axis=0)

            if (default_config.alg == 'MARC' and all([ob['unary_tensor'].any() for ob in obs + next_obs])):
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

            elif default_config.alg == 'MAAC':
                replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)


            episode_reward_total += rewards.sum()
            obs = next_obs
            t += 1
            if (len(replay_buffer) >= hyper_config.batch_size and
                    (t % default_config.steps_per_update) ==0):

                if default_config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(default_config.num_updates):
                    sample = replay_buffer.sample(hyper_config.batch_size,
                                                  to_gpu=default_config.use_gpu, norm_rews=hyper_config.norm_rews)
                    model.update_critic(sample)
                    model.update_policies(sample)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
            if dones.all() == True:
                print('done with time step', et_i)
                break
        steps += et_i
        print("%s - %s - Episodes %i (%is) of %i - Reward %.2f" % (default_config.env_id, default_config.random_seed, ep_i + 1, steps,
                                        default_config.n_episodes,  episode_reward_total))
        l_rewards.append(episode_reward_total)
        epymarl_rewards.append(episode_reward_total)
        # check if it in average was the best model so far


        if steps % default_config.step_interval_log == 0:
            wandb.log({'mean_return': np.mean(epymarl_rewards)})
            epymarl_rewards.clear()


        if ep_i % default_config.test_interval ==0:
            print('ep_i', ep_i)
            print('This should be 0:', ep_i % default_config.test_interval )
            model.prep_rollouts(device='cpu')
            l_ep_rew = []
            for eval_ep_i in range(default_config.test_n_episodes):
                ep_rew = 0

                obs = env.reset()
                for t_i in range(1, default_config.episode_length+1):

                    # rearrange observations to be per agent, and convert to torch Variable

                    if default_config.alg == 'MARC':
                        obs = [np.expand_dims(ob['image'].flatten(), axis=0) for ob in obs]
                        torch_obs = [Variable(torch.Tensor(agent_obs[i]),
                                              requires_grad=False)
                                     for i in range(model.n_agents)]
                        # get actions as torch Variables
                        torch_actions = model.step(torch_obs, explore=False)
                        # convert actions to numpy arrays
                        actions = [np.argmax(ac.data.numpy().flatten()) for ac in torch_actions]
                    else:
                        if env.grid_observation:
                            obs = tuple([obs[:, i][0]['image'].flatten() for i in range(model.nagents)])
                            obs = np.vstack(obs)
                            obs = np.expand_dims(obs, axis=0)
                        torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                              requires_grad=False)
                                     for i in range(model.nagents)]
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
                print("Episode %i of %i - %.2f" % (eval_ep_i + 1, default_config.test_n_episodes, ep_rew))

            avg_eval_rew = sum(l_ep_rew) / default_config.test_n_episodes
            print("Average eval reward: {}".format(avg_eval_rew))
            wandb.log({'avg_eval_return': avg_eval_rew})

    env.close()
    wandb.finish()

def make_parallel_MAAC_env(args):
    def get_env_fn(rank):
        def init_env():
            env = make_env(args)
            env.agents = [Agent() for _ in range(args.player)]
            # env.grid_observation = args.grid_observation
            # env.seed(args.random_seed + rank * 1000)
            np.random.seed(args.random_seed + rank * 1000)
            return env
        return init_env
    # if default_config.n_rollout_threads == 1:
    return DummyVecEnv([get_env_fn(0)])
    # else:
    #     return SubprocVecEnv([get_env_fn(i) for i in range(default_config.n_rollout_threads)])

def make_env(default_config):
    from lbforaging.foraging import ForagingEnv
    env = ForagingEnv(
        players=default_config.player,
        max_player_level=2,
        field_size=(15,15),
        max_food= default_config.max_food,
        grid_observation=False,
        sight=15,
        max_episode_steps=default_config.episode_length,
        force_coop=False,
        keep_food=True,
    )
    return env


if __name__ == '__main__':
    sweep_config = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "mean_return"},
        "parameters": {
            "critic_hidden_dim": {'values':[64, 128]},
            "pol_hidden_dim": {'values':[64, 128]},
            "pi_lr": {'values':[0.0005, 0.001]},
            "q_lr": {'values':[0.0005, 0.001]},
            'tau': {'values':[0.001, 0.01]},
            'batch_size':{'values': [512, 1024]},
            'attend_heads': {'values':[2, 4, 6]},
            'norm_rews': {'value': True},
            'force_coop': {'value' : False},
            'episode_length': {'value': 50},
            'keep_food': {'value' : True},
            'grid_observation': {'value': False},
            # "reward_standardization": [True, False],
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="MARC")
    wandb.agent(sweep_id, function=run, count=1)
    # id : 3pg1uu5u


