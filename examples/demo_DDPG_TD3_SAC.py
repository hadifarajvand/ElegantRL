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


def train_ddpg_td3_sac_for_pendulum(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.break_step = int(2e5)  # break training if 'total_step > break_step'
    args.net_dims = [64, 64]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 256
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step // 2

    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 5e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.gpu_id = GPU_ID
    args.num_workers = 8
    train_agent(args=args, if_single_process=False)
    """
-2000 < -1200 < -200 < -80
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  3.20e+03       4 |-1463.02   44.1    200     0 |   -7.02   2.31 -21.30
0  2.40e+04      34 |-1049.32   21.9    200     0 |   -7.57   1.80-103.58
0  4.48e+04      61 | -163.25   62.2    200     0 |   -2.42   1.36-103.66
0  6.56e+04      88 | -199.53  126.3    200     0 |   -0.96   1.40 -71.96
| UsedTime:     110 | SavedDir: ./Pendulum_DDPG_0
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  4.00e+02       1 |-1211.60    4.7    200     0 |   -8.24   7.17  -5.75
0  2.04e+04      58 | -207.35  138.9    200     0 |   -0.91   2.25 -45.49
0  4.04e+04     117 |  -85.54   71.5    200     0 |   -0.95   1.04 -17.13
| UsedTime:     146 | SavedDir: ./Pendulum_TD3_0

################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  1.60e+03       2 | -501.07  112.2    200     0 |   -2.87   0.00   0.00
0  2.16e+04      15 | -650.15   53.9    200     0 |   -3.19   1.53  -1.44
0  4.16e+04      29 | -825.14    9.0    200     0 |   -4.33   0.06  -7.26
0  6.16e+04      44 | -747.52    7.9    200     0 |   -3.39   0.03 -15.75
0  8.16e+04      60 | -538.42    8.0    200     0 |   -2.70   0.02 -26.40
0  1.02e+05      77 | -198.29    1.7    200     0 |   -1.32   0.03 -35.16
0  1.22e+05      94 |  -85.85   39.1    200     0 |   -0.00   0.07 -38.01
0  1.42e+05     114 |  -39.97   68.4    200     0 |   -0.97   0.18 -38.21
0  1.62e+05     134 |  -81.10   32.1    200     0 |   -0.00   0.27 -35.48
0  1.82e+05     154 |  -60.41   59.1    200     0 |   -0.75   0.38 -31.22
| UsedTime:     175 | SavedDir: ./Pendulum_TD3_0
    """


def train_ddpg_td3_sac_for_pendulum_vec_env(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name
    num_envs = 8

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'Pendulum',  # Apply torque on the free end to swing a pendulum into an upright position
        'max_step': 200,  # the max step number of an episode.
        'state_dim': 3,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': 1,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=PendulumEnv(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.net_dims = [128, 64]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 256  # vectorized env need a larger batch_size
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step // 8

    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 4e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_per_step = int(1e4)
    args.break_step = int(8e4)  # break training if 'total_step > break_step'

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)
    """
-2000 < -1200 < -200 < -80
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
0  2.00e+02       7 |-1044.31    0.0    200     0 |   -7.03  19.73  -6.14
0  4.20e+03      46 | -143.11    0.0    200     0 |   -2.82   2.59 -47.02
0  8.20e+03      85 | -132.30    0.0    200     0 |   -1.73   3.48 -34.18
| UsedTime:     105 | SavedDir: ./Pendulum_TD3_0
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
3  2.00e+02       2 |-1229.57   81.1    200     0 |   -5.87   6.28  -9.39   0.34
3  4.20e+03      42 | -194.66  136.3    200     0 |   -3.46   0.82 -91.20   0.36
3  8.20e+03      83 |  -94.45   57.3    200     0 |   -0.62   0.67 -57.82   0.29
| UsedTime:     102 | SavedDir: ./Pendulum_ModSAC_0
    """


def train_ddpg_td3_sac_for_lunar_lander_continuous(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name):

    import gymnasium as gym
    agent_class = [AgentTD3, AgentSAC, AgentModSAC][DRL_ID]  # DRL algorithm name
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {'env_name': 'LunarLanderContinuous-v3',
                'num_envs': 1,
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False}
    get_gym_env_args(env=gym.make('LunarLanderContinuous-v3'), if_print=True)  # return env_args

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

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)
    """
-1500 < -200 < 200 < 290
################################################################################
ID     Step    Time |    avgR   stdR   avgS  stdS |    expR   objC   objA   etc.
3  3.60e+04     155 |   55.10  115.1    669   378 |   -0.00   0.65   2.21   0.03
3  7.60e+04     241 |  212.24   65.0    403   229 |    0.05   0.80  11.81   0.06
3  1.16e+05     350 |   49.57   78.3    308   341 |    0.01   0.57  14.12   0.04
3  1.56e+05     500 |   97.97  103.7    754   302 |    0.13   0.62  12.42   0.04
3  1.96e+05     596 |  134.18  133.3    562   328 |    0.14   0.58  10.31   0.03
3  2.36e+05     690 |  247.05   26.1    298    96 |    0.21   0.52   8.43   0.03
3  2.76e+05     801 |  148.64  111.6    218   114 |    0.21   0.61   8.90   0.03
3  3.16e+05     920 |  174.72  138.2    353   235 |    0.36   0.58   9.07   0.03
3  3.56e+05    1031 |  210.69   89.8    329   234 |    0.27   0.61   8.85   0.03
3  3.96e+05    1139 |  260.68   25.7    275    86 |    0.34   0.61   9.23   0.02
| UsedTime:    1147 | SavedDir: ./LunarLanderContinuous-v3_ModSAC_0
    """


def train_ddpg_td3_sac_for_lunar_lander_continuous_vec_env(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name
    num_envs = 8

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

    args.gpu_id = GPU_ID
    args.num_workers = 4
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
| UsedTime:     627 | SavedDir: ./LunarLanderContinuous-v3_ModSAC_0
    """


def train_ddpg_td3_sac_for_bipedal_walker_env(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'BipedalWalker-v3',
        'num_envs': 1,
        'max_step': 1600,
        'state_dim': 24,
        'action_dim': 4,
        'if_discrete': False,
    }
    get_gym_env_args(env=gym.make('BipedalWalker-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.net_dims = [256, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards

    args.horizon_len = args.max_step // 32
    args.buffer_init_size = args.max_step // 32
    args.repeat_times = 0.5  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 1e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.soft_update_tau = 5e-3
    args.buffer_size = int(2e6)
    args.break_step = int(5e6)  # break training if 'total_step > break_step'

    args.policy_noise_std = 0.10  # standard deviation of exploration noise
    args.explore_noise_std = 0.05  # standard deviation of exploration noise

    args.eval_times = 8
    args.eval_per_step = int(1e4)

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)
    """
-200 < -100 < 300 < 320
    """


def train_ddpg_td3_sac_for_bipedal_walker_vec_env(agent_class):
    assert agent_class in {AgentDDPG, AgentTD3, AgentSAC, AgentModSAC}  # DRL algorithm name
    num_envs = 8

    import gymnasium as gym
    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        'env_name': 'BipedalWalker-v3',
        'max_step': 1600,
        'state_dim': 24,
        'action_dim': 4,
        'if_discrete': False,

        'num_envs': num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    get_gym_env_args(env=gym.make('BipedalWalker-v3'), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `erl_config.py Arguments()` for hyperparameter explanation
    args.net_dims = [256, 128]  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 512
    args.gamma = 0.99  # discount factor of future rewards

    args.horizon_len = args.max_step // 32
    args.buffer_init_size = args.max_step // 32
    args.repeat_times = 0.5  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.reward_scale = 2 ** 0
    args.learning_rate = 1e-4
    args.state_value_tau = 0  # the tau of normalize for value and state `std = (1-std)*std + tau*std`
    args.soft_update_tau = 5e-3
    args.buffer_size = int(2e6)
    args.break_step = int(5e6)  # break training if 'total_step > break_step'

    args.policy_noise_std = 0.10  # standard deviation of exploration noise
    args.explore_noise_std = 0.05  # standard deviation of exploration noise

    args.eval_times = 8
    args.eval_per_step = int(1e4)

    args.gpu_id = GPU_ID
    args.num_workers = 4
    train_agent(args=args, if_single_process=False)
    """
-200 < -100 < 300 < 320
    """


if __name__ == '__main__':
    Parser = ArgumentParser(description='ArgumentParser for ElegantRL')
    Parser.add_argument('--gpu', type=int, default=0, help='GPU device ID for training')
    Parser.add_argument('--drl', type=int, default=3, help='RL algorithms ID for training')
    Parser.add_argument('--env', type=str, default='1', help='the environment ID for training')

    Args = Parser.parse_args()
    GPU_ID = Args.gpu
    DRL_ID = Args.drl
    ENV_ID = Args.env

    AgentClassList = [AgentTD3, AgentSAC, AgentModSAC, AgentDDPG]
    AgentClass = AgentClassList[DRL_ID]  # DRL algorithm name
    if ENV_ID in {'0', 'pendulum'}:
        train_ddpg_td3_sac_for_pendulum(agent_class=AgentClass)
    elif ENV_ID in {'1', 'pendulum_vec'}:
        train_ddpg_td3_sac_for_pendulum_vec_env(agent_class=AgentClass)
    elif ENV_ID in {'2', 'lunar_lander_continuous'}:
        train_ddpg_td3_sac_for_lunar_lander_continuous(agent_class=AgentClass)
    elif ENV_ID in {'3', 'lunar_lander_continuous_vec'}:
        train_ddpg_td3_sac_for_lunar_lander_continuous_vec_env(agent_class=AgentClass)
    elif ENV_ID in {'4', 'lunar_lander_continuous'}:
        train_ddpg_td3_sac_for_bipedal_walker_env(agent_class=AgentClass)
    elif ENV_ID in {'5', 'lunar_lander_continuous_vec'}:
        train_ddpg_td3_sac_for_bipedal_walker_vec_env(agent_class=AgentClass)
    else:
        print(f'ENV_ID not match. type(ENV_ID) is str not {type(ENV_ID)}', flush=True)
