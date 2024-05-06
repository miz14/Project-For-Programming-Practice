import random
import time
import gym
import numpy as np
import torch
from ppo import Agent, make_env, Args
from moviepy import *
import math

args = Args
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
)
#Выборочное среднее: 452.8166809082031; std: 107.27496337890625; var 11507.9169921875
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.05): 424.88046978939065, 480.7528920270156
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.1): 429.48366690220405, 476.1496949142022

#Выборочное среднее: 411.3333435058594; std: 88.36904907226562; var 7809.08837890625
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.05): 388.32055157852193, 434.3461354331968
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.1): 392.11249089724936, 430.5541961144694
checkpoints = [
    torch.load('breakout/DPG/models/agent_checkpoint_859.pth'),
    torch.load('breakout/DPG/models/agent_checkpoint_full.pth'),
]

agent = Agent(envs).to(device)

n_x = 60

for check in checkpoints:

    agent.load_state_dict(check['agent_state_dict'])
    next_obs, _ = envs.reset()
    next_done = False

    rewards = []
    while True:
        action, logprob, _, value = agent.get_action_and_value(torch.Tensor(next_obs).to(device))
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        if "final_info" in infos:
            if infos['final_info'][0]['lives'] == 0:
                next_obs, _ = envs.reset()
                rewards.append(infos['final_info'][0]['episode']['r'])
                print(rewards)
                if len(rewards) == 1:
                    err
                    break

    print(sorted(rewards))

    a = np.mean(rewards)
    alpha = 0.05
    S = np.std(rewards)
    S_2 = np.var(rewards)
    n_free = n_x - 1
    t_student_0_05 = 2.0003
    t_student_0_1 = 1.6707

    print(f'Выборочное среднее: {a}; std: {S}; var {S_2}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.05): {a - S/math.sqrt(n_free)* t_student_0_05}, {a + S / math.sqrt(n_free)* t_student_0_05}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.1): {a - S/math.sqrt(n_free)* t_student_0_1}, {a + S / math.sqrt(n_free)* t_student_0_1}')

