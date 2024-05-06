import random
import matplotlib.pyplot as plt
import time
import gymnasium as gym
import numpy as np
import torch
import math

from dqn_arguments import Args as args
from dqn import make_env, QNetwork, linear_schedule


run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
best_reward = 20
total_reward = []

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)

#Выборочное среднее: 254.93333435058594; std: 57.21709060668945; var 3273.795654296875
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.05): 240.03303821021922, 269.83363049095266
#Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.1): 242.48823873407306, 267.3784299670988

checkpoints = [
    torch.load('breakout/VB/models/target_network_full.pth')
    # torch.load('breakout/VB/models/target_network_checkpoint_[754.].pth')
    # torch.load('breakout/VB/models/target_network_checkpoint_[448.].pth'),
]

q_network = QNetwork(envs).to(device)

n_x = 60

for check in checkpoints:
    q_network.load_state_dict(check['target_network_state_dict'])
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = False
    rewards = []
    action_stop = False
    while True:
        q_values = q_network(torch.Tensor(next_obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        if action_stop:
            next_obs, reward, terminations, truncations, infos = envs.step([2])
        else:
            next_obs, reward, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            if infos['final_info'][0]['lives'] == 0:
                print(rewards)
                rewards.append(infos['final_info'][0]['episode']['r'][0])
                next_obs, _ = envs.reset()
                action_stop = False
                if len(rewards) == n_x:
                    break
        elif infos['episode_frame_number'][0] > 20000 and not action_stop:
            action_stop = True

                    

    print(sorted(rewards))
    a = np.mean(rewards)
    S = np.std(rewards)
    S_2 = np.var(rewards)
    n_free = n_x - 1
    t_student_0_05 = 2.0003
    t_student_0_1 = 1.6707

    print(f'Выборочное среднее: {a}; std: {S}; var {S_2}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.05): {a - S/math.sqrt(n_free)* t_student_0_05}, {a + S / math.sqrt(n_free)* t_student_0_05}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии (I=0.1): {a - S/math.sqrt(n_free)* t_student_0_1}, {a + S / math.sqrt(n_free)* t_student_0_1}')