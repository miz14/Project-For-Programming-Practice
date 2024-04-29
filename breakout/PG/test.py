import random
import time
import gymnasium as gym
import numpy as np
import torch
from ppo import Agent, make_env, Args
from moviepy import *

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
checkpoint = torch.load('models/agent_checkpoint_full.pth')

agent = Agent(envs).to(device)

agent.load_state_dict(checkpoint['agent_state_dict'])

next_obs, _ = envs.reset(seed=args.seed)
next_done = False
total_reward = 0
while True:
    action, logprob, _, value = agent.get_action_and_value(torch.Tensor(next_obs).to(device))
    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
    total_reward += reward[0]
    if "final_info" in infos:
        for info in infos["final_info"]:
            if info['lives'] == 0:
                next_obs, _ = envs.reset(seed=args.seed)
                print(total_reward)
                total_reward = 0