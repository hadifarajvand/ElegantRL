from argparse import ArgumentParser

try:
    from ..elegantrl import Config
    from ..elegantrl import train_agent
    from ..elegantrl import get_gym_env_args
    from ..elegantrl.agents import AgentDDPG, AgentTD3
    from ..elegantrl.agents import AgentSAC, AgentModSAC
except ImportError or ModuleNotFoundError:
    from elegantrl import Config
    from elegantrl import train_agent
    from elegantrl import get_gym_env_args
    from elegantrl.agents import AgentDDPG, AgentTD3
    from elegantrl.agents import AgentSAC, AgentModSAC


def train_ddpg_td3_sac_for_lunar_lander_continuous(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name):

    import gymnasium as gym
    agent_class = [AgentTD3, AgentSAC, AgentModSAC][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLanderContinuous-v2',
                'num_envs': 1,
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False}
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v2'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.net_dims = [256, 256]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 1024
    args.gamma = 0.99  # discount factor of future rewards
    args.horizon_len = args.max_step // 2
    args.repeat_times = 1  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 1e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_times = 32
    args.eval_per_step = int(4e4)
    args.break_step = int(4e5)  # break training if 'total_step > break_step'

    args.if_use_per = True
    args.per_alpha = 0.6  # see elegantrl/train/replay_buffer.py self.per_alpha = getattr(args, 'per_alpha', 0.6)
    args.per_beta = 0.4  # see elegantrl/train/replay_buffer.py self.per_beta = getattr(args, 'per_beta', 0.4)
    args.buffer_size = int(4e5)  # PER can handle larger buffer_size
    args.repeat_times = 0.5  # PER don't need a large repeat_times

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)
    """
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
2  4.00e+03      73 |  -53.40  119.4    733   226 |   -0.90   1.21  -2.89   0.28
2  4.40e+04     186 |   16.26  104.6    862   239 |    0.00   0.81   4.85   0.06
2  8.40e+04     254 |  126.60  160.5    485   272 |    0.09   0.68   9.82   0.04
2  1.24e+05     393 |  110.49   96.4    826   221 |   -0.03   0.71  11.30   0.03
2  1.64e+05     472 |  141.29  146.3    561   271 |    0.08   0.52   9.66   0.04
2  2.04e+05     572 |  151.07  124.5    364   302 |    0.12   0.55  10.28   0.03
2  2.44e+05     668 |   55.92  104.6    149   110 |    0.01   0.53  11.39   0.03
2  2.84e+05     798 |  207.54   94.2    484   254 |    0.09   0.55  12.60   0.03
2  3.24e+05     902 |  179.12  119.1    512   316 |    0.13   0.60  11.21   0.02
2  3.64e+05     993 |  250.26   21.2    351   170 |    0.41   0.56  10.92   0.02
| UsedTime:    1082 | SavedDir: ./LunarLanderContinuous-v2_ModSAC_0

-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
3  4.00e+03      20 | -237.07   67.0    233   105 |   -0.71   1.37  -1.28
3  4.40e+04      77 |  -22.37   41.1    137    52 |    0.09   1.60   6.54
3  8.40e+04     140 | -235.39   99.6    165    42 |   -0.46   1.46  10.86
3  1.24e+05     254 | -151.96   95.3    715   327 |   -0.21   1.98  12.80
3  1.64e+05     344 |  -33.57   33.2    916   266 |   -0.02   1.49  13.89
3  2.16e+05     425 |  -34.89   50.6    883   298 |    0.00   1.31  15.08
3  2.56e+05     488 |  -44.32   41.3    857   281 |    0.04   1.30  15.90
3  2.96e+05     549 |  -51.70   59.5    847   264 |   -0.03   1.05  16.05
3  3.36e+05     624 |  -16.65   29.0    891   293 |   -0.02   0.88  14.85
3  3.76e+05     682 |   -6.47   83.7    806   322 |    0.01   0.84  13.59
| UsedTime:     683 | SavedDir: ./LunarLanderContinuous-v2_TD3_0    # need to fine-tune
    """


def train_ddpg_td3_sac_for_lunar_lander_continuous_vec_env(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name
    num_envs = 4

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'LunarLanderContinuous-v3',
        'max_step': 1000,
        'state_dim': 8,
        'action_dim': 2,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.net_dims = [256, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards

    args.horizon_len = args.max_step // 4
    args.buffer_init_size = args.max_step * 2
    args.repeat_times = 2  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** -1
    args.learning_rate = 1e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.soft_update_tau = 5e-3
    args.buffer_size = int(2e6)
    args.break_step = int(5e6)  # break training if 'total_step > break_step'

    args.policy_noise_std = 0.10  # standard deviation of exploration noise
    args.explore_noise_std = 0.05  # standard deviation of exploration noise

    args.eval_times = 32
    args.eval_per_step = int(1e5)

    args.if_use_per = True
    args.per_alpha = 0.6  # see elegantrl/train/replay_buffer.py self.per_alpha = getattr(args, 'per_alpha', 0.6)
    args.per_beta = 0.4  # see elegantrl/train/replay_buffer.py self.per_beta = getattr(args, 'per_beta', 0.4)
    args.buffer_size = int(4e5)  # PER can handle larger buffer_size
    args.repeat_times = 0.5  # PER don't need a large repeat_times

    args.gpu_id = GPU_ID
    args.num_workers = 2
    train_agent(args=args, if_single_process=False)
    """
-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
2  2.00e+03      47 | -168.82  115.1    635   361 |   -0.78   1.42  -4.94   0.28
2  2.20e+04     106 |  -64.67   26.2   1000     0 |   -0.05   0.65  -2.20   0.04
2  4.20e+04     166 |  -26.96   17.3    678   443 |   -0.03   0.52   2.57   0.04
2  6.20e+04     229 |   74.15  117.1    421   380 |   -0.03   0.48   6.46   0.04
2  8.20e+04     291 |  157.59  133.7    421   357 |   -0.00   0.47   8.84   0.04
2  1.02e+05     349 |  216.84   88.3    369   279 |    0.17   0.47   9.58   0.03
2  1.22e+05     402 |  167.54  115.0    219   143 |    0.17   0.57  11.03   0.04
2  1.42e+05     466 |  241.06   60.2    299   141 |    0.25   0.54  11.62   0.03
2  1.62e+05     527 |  257.67   39.7    243   108 |    0.21   0.49  11.09   0.02
2  1.82e+05     598 |  242.47   56.4    441   252 |    0.54   0.57  10.63   0.02
| UsedTime:     627 | SavedDir: ./LunarLanderContinuous-v2_ModSAC_0
    """


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=2, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='1', help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    AgentClassList = [AgentTD3, AgentSAC, AgentModSAC, AgentDDPG]
    AgentClass = AgentClassList[DRL_ID]  # DRL algorithm name
    if ENV_ID in {'0', 'lunar_lander_continuous'}:
        train_ddpg_td3_sac_for_lunar_lander_continuous(agent_class=AgentClass)
    elif ENV_ID in {'1', 'lunar_lander_continuous_vec'}:
        train_ddpg_td3_sac_for_lunar_lander_continuous_vec_env(agent_class=AgentClass)
    else:
        print(f'ENV_ID not match. type(ENV_ID) is str not {type(ENV_ID)}', flush=True)
