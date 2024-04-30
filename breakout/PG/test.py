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

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
)

checkpoints = [
    torch.load('breakout/PG/models/agent_checkpoint[41.].pth'),
    torch.load('breakout/PG/models/agent_checkpoint_full.pth'),
]

agent = Agent(envs).to(device)

n_x = 60

for check in checkpoints:

    agent.load_state_dict(check['agent_state_dict'])
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = False
    total_reward = 0

    rewards = []
    while True:
        action, logprob, _, value = agent.get_action_and_value(torch.Tensor(next_obs).to(device))
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        total_reward += reward[0]
        if "final_info" in infos:
            if infos['final_info'][0]['lives'] == 0:
                next_obs, _ = envs.reset(seed=args.seed)
                rewards.append(total_reward)
                total_reward = 0
                if len(rewards) == 1:
                    break
    err
    print()
    print(sorted(rewards))

    a = np.mean(rewards)
    alpha = 0.05
    S = np.std(rewards)
    S_2 = np.var(rewards)
    n_free = n_x - 1
    t_student = 2.000

    print(f'Выборочное среднее: {a}; std: {S}; var {S_2}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии: {a - S/math.sqrt(n_free)* t_student}, {a + S / math.sqrt(n_free)* t_student}')

