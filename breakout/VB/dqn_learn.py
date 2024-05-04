import random
import matplotlib.pyplot as plt
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from dqn_arguments import Args as args
from dqn import make_env, QNetwork, linear_schedule
from stable_baselines3.common.buffers import ReplayBuffer



run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
best_reward = 20
total_reward = []

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)

q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())

print(envs.single_observation_space)

rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    optimize_memory_usage=True,
    handle_timeout_termination=False,
)
start_time = time.time()

# TRY NOT TO MODIFY: start the game
obs, _ = envs.reset(seed=args.seed)
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_network(torch.Tensor(obs).to(device))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

    # TRY NOT TO MODIFY: execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                episode_reward = info['episode']['r']
                if episode_reward >= best_reward:
                    best_reward = episode_reward
                    checkpoint = {
                                        'target_network_state_dict': target_network.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                }
                    torch.save(checkpoint, f'models/target_network_checkpoint_{episode_reward}.pth')

                total_reward.append(episode_reward)
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > args.learning_starts:
        if global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            if global_step % 100 == 0:
                print("SPS:", int(global_step / (time.time() - start_time)))

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update target network
        if global_step % args.target_network_frequency == 0:
            for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                target_network_param.data.copy_(
                    args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                )

if args.save_model:
    torch.save(checkpoint, f'models/target_network_full.pth')
envs.close()
plt.plot(total_reward)
plt.title('Зависимсоть вознаграждения')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()