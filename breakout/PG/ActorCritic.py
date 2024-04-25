import torch
import torch.nn as nn
import torch.nn.functional as F

# class ActorCriticModel(nn.Module):
#     def __init__(self, n_state, n_action, n_hidden):
#         super(ActorCriticModel, self).__init__()
#         self.conv2d_1 = nn.Conv2d(n_state[0], 16, 16, 8) 

#         # self.conv2d_2 = nn.Conv2d(32, 16, 8, 4)


#         self.fc1 = nn.Linear(16 * 8 * 8, n_hidden[0])
#         self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
#         self.action = nn.Linear(n_hidden[1], n_action)
#         self.value = nn.Linear(n_hidden[1], 1)

#     def forward(self, x):
#         x = torch.Tensor(x)
#         x = F.relu(self.conv2d_1(x))
#         # print(x.shape)
#         # err
#         # x = F.relu(self.conv2d_2(x))
#         x = x.view(-1)
#         x = self.fc1(x)
#         x = F.relu(self.fc2(x))
#         action_probs = F.softmax(self.action(x), dim=-1)
#         state_value = self.value(x)
#         return action_probs, state_value
    
# class PolicyNetwork():
#     def __init__(self, n_state, n_action, n_hidden = [64, 32], lr=0.001):
#         self.model = ActorCriticModel(n_state, n_action, n_hidden)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
#         self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

#     def predict(self, s): #отдает распредление
#         return self.model(torch.tensor(s).float())

#     def update(self, returns, log_probs, state_values): #self, веса, логарифмы от вероятности
#         loss = 0
#         for log_prob, value, Gt in zip(log_probs, state_values, returns):
#             advantage = Gt - value.item()
#             policy_loss = -log_prob * advantage
#             value_loss = F.smooth_l1_loss(value, Gt)
#             loss += policy_loss + value_loss

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#     def get_action(self, s): #состояние -> розыгрыш действия -> поиск логарифма вероятности действия
#         action_probs, state_value = self.predict(s)
#         action = torch.multinomial(action_probs, 1).item()
#         log_prob = torch.log(action_probs[action])
#         return action, log_prob, state_value

# def actor_critic(env, estimator, n_episode, gamma=1.0):
#     total_reward_episode = [0] * n_episode
#     for episode in range(n_episode):
#         log_probs = []
#         rewards = []
#         state_values = []

#         state, info = env.reset()
#         while True:
#             # one_hot_state = [0] * 48 #сделали one hot сосотяний
#             # one_hot_state[state] = 1
            

#             action, log_prob, state_value = estimator.get_action(state)
#             next_state, reward, is_done, t, info = env.step(action)
            
#             total_reward_episode[episode] += reward
#             log_probs.append(log_prob)
#             rewards.append(reward)
#             state_values.append(state_value)

#             if is_done:
#                 returns = []
#                 Gt = 0
#                 pw = 0
#                 for reward in rewards[::-1]:
#                     Gt += gamma ** pw * reward
#                     pw += 1
#                     returns.append(Gt)
#                 returns = returns[::-1]
#                 returns = torch.tensor(returns)
#                 returns = (returns - returns.mean()) / (returns.std() + 1e-9)
#                 estimator.update(returns, log_probs, state_values)

#                 if total_reward_episode[episode] >= 100:
#                     estimator.scheduler.step()
#                 break

#             else:
#                 state = next_state
#         print(episode, total_reward_episode[episode])
#     return total_reward_episode









import torch.nn as nn
import torch


class PolicyNetworkModel(nn.Module):
    def __init__(self, n_state, n_action):
        super(PolicyNetworkModel, self).__init__()

        self.relu = nn.ReLU()
        self.conv2d_1 = nn.Conv2d(n_state[0], 32, 8, 4) 

        self.conv2d_2 = nn.Conv2d(32, 16, 8, 4)

        self.linear_1 = nn.Linear(16 * 4 * 4, 64)

        self.linear_2 = nn.Linear(64, n_action)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # [4, 80, 80]
        x = self.relu(self.conv2d_1(x)) # [16, 20, 20]
        # print(x.shape)
        x = self.relu(self.conv2d_2(x)) # [32, 4, 4]
        x = x.view(-1)
        x = self.relu(self.linear_1(x))
        return self.softmax(self.relu(self.linear_2(x)))



class PolicyNetwork():
    def __init__(self, n_state, n_action, lr=0.001):
        
        self.model = PolicyNetworkModel(n_state, n_action)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
    def predict(self, s):
        """
        Вычисляет вероятности действий в состоянии s,
            применяя обученную модель
        @param s: входное состояние
        @return: предсказанная стратегия
        """
        return self.model(torch.tensor(s).float())

    def update(self, returns, log_probs):
        """
        Обновляет веса сети стратегии на основе обучающих примеров
        @param returns: доход (накопительное вознаграждение) на каждом
        шаге эпизода
        @param log_probs: логарифмы вероятностей на каждом шаге
        """
        policy_gradient = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_gradient.append(-log_prob * Gt)
            
        loss = torch.stack(policy_gradient).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
        Предсказывает стратегию, выбирает действие и вычисляет логарифм
        его вероятности
        @param s: входное состояние
        @return: выбранное действие и логарифм его вероятности
        """
        probs = self.predict(s)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        load = torch.load(path)
        self.model.load_state_dict(load['model_state_dict'])
        self.optimizer.load_state_dict(load['optimizer_state_dict'])
        self.model.eval()




# def reinforce(env, estimator, n_episode, gamma=1.0):
#     total_reward_episode = [0] * n_episode
#     """
#     Алгоритм REINFORCE
#     @param env: имя окружающей среды Gym
#     @param estimator: сеть, аппроксимирующая стратегию
#     @param n_episode: количество эпизодов
#     @param gamma: коэффициент обесценивания
#     """
#     for episode in range(n_episode):
#         log_probs = []
#         rewards = []
#         state, _, info = env.reset()
#         while True:
#             action, log_prob = estimator.get_action(state)
#             next_state, reward, is_done, t, _, info = env.step(action)
#             total_reward_episode[episode] += reward
    
#             log_probs.append(log_prob)
#             rewards.append(reward)
#             if is_done or t:
#                 returns = []
#                 Gt = 0
#                 pw = 0
#                 for reward in rewards[::-1]:
#                     Gt += gamma ** pw * reward
#                     pw += 1
#                     returns.append(Gt)
#                 returns = returns[::-1]
#                 returns = torch.tensor(returns)
#                 returns = (returns - returns.mean()) / (
#                         returns.std() + 1e-9)
#                 estimator.update(returns, log_probs)
#                 print('Эпизод: {}, полное вознаграждение: {}'.
#                     format(episode, total_reward_episode[episode]))
#                 break
#             state = next_state
#         # print(episode, total_reward_episode[episode])
def reinforce(env, estimator, n_episode, gamma=1.0):
    total_reward_episode = [0] * n_episode
    for episode in range(n_episode):
        log_probs = []
        rewards = []
        state, _ = env.reset()

        q = 0
        while True:
            action, log_prob = estimator.get_action(state)
            next_state, reward, is_done, t, info  = env.step(action)

            total_reward_episode[episode] += reward #глобальная переменная
            log_probs.append(log_prob)
            rewards.append(reward)
            # if reward > 1e-4:
            #     print("rew", rewards)
            q += 1

            if is_done: #терминальное состояние
                returns = [] #считаем Gt, раскручиваем траеткторию в обратном направлении
                Gt = 0
                pw = 0

                for reward in rewards[::-1]: #обратное направление
                    Gt += gamma ** pw * reward
                    pw += 1
                    returns.append(Gt)

                returns = returns[::-1]
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + 1e-9) #нормирование (+ чтоб 0 не было)

                estimator.update(returns, log_probs)
                print(f'Episode: {episode}, total reward: {total_reward_episode[episode]}')
                break

            state = next_state
    return total_reward_episode

# import os
# import numpy as np
# import torch as T
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions.categorical import Categorical
# import matplotlib.pylab as plt

# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []

#         self.batch_size = batch_size

#     def generate_batches(self):
#         n_states = len(self.states)
#         batch_start = np.arange(0, n_states, self.batch_size)
#         indices = np.arange(n_states, dtype=np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i+self.batch_size] for i in batch_start]

#         return np.array(self.states),\
#                 np.array(self.actions),\
#                 np.array(self.probs),\
#                 np.array(self.vals),\
#                 np.array(self.rewards),\
#                 np.array(self.dones),\
#                 batches

#     def store_memory(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)

#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []

# class ActorNetwork(nn.Module):
#     def __init__(self, n_actions, input_dims, alpha,
#             fc1_dims=256, fc2_dims=256, chkpt_dir='ppo'):
#         super(ActorNetwork, self).__init__()

#         self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
#         self.actor = nn.Sequential(
#                 nn.Linear(*input_dims, fc1_dims),
#                 nn.ReLU(),
#                 nn.Linear(fc1_dims, fc2_dims),
#                 nn.ReLU(),
#                 nn.Linear(fc2_dims, n_actions),
#                 nn.Softmax(dim=-1)
#         )

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, state):
#         dist = self.actor(state)
#         dist = Categorical(dist)

#         return dist

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))

# class CriticNetwork(nn.Module):
#     def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
#             chkpt_dir='ppo'):
#         super(CriticNetwork, self).__init__()

#         self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
#         self.critic = nn.Sequential(
#                 nn.Linear(*input_dims, fc1_dims),
#                 nn.ReLU(),
#                 nn.Linear(fc1_dims, fc2_dims),
#                 nn.ReLU(),
#                 nn.Linear(fc2_dims, 1)
#         )

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, state):
#         value = self.critic(state)

#         return value

#     def save_checkpoint(self):
#         T.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(T.load(self.checkpoint_file))

# class Agent:
#     def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
#             policy_clip=0.2, batch_size=64, n_epochs=10):
#         self.gamma = gamma
#         self.policy_clip = policy_clip
#         self.n_epochs = n_epochs
#         self.gae_lambda = gae_lambda

#         self.actor = ActorNetwork(n_actions, input_dims, alpha)
#         self.critic = CriticNetwork(input_dims, alpha)
#         self.memory = PPOMemory(batch_size)

#     def remember(self, state, action, probs, vals, reward, done):
#         self.memory.store_memory(state, action, probs, vals, reward, done)

#     def save_models(self):
#         print('... saving models ...')
#         self.actor.save_checkpoint()
#         self.critic.save_checkpoint()

#     def load_models(self):
#         print('... loading models ...')
#         self.actor.load_checkpoint()
#         self.critic.load_checkpoint()

#     def choose_action(self, observation):
#         state = T.tensor([observation], dtype=T.float).to(self.actor.device)

#         dist = self.actor(state)
#         value = self.critic(state)
#         action = dist.sample()

#         probs = T.squeeze(dist.log_prob(action)).item()
#         action = T.squeeze(action).item()
#         value = T.squeeze(value).item()

#         return action, probs, value

#     def learn(self):
#         for _ in range(self.n_epochs):
#             state_arr, action_arr, old_prob_arr, vals_arr,\
#             reward_arr, dones_arr, batches = \
#                     self.memory.generate_batches()

#             values = vals_arr
#             advantage = np.zeros(len(reward_arr), dtype=np.float32)

#             for t in range(len(reward_arr)-1):
#                 discount = 1
#                 a_t = 0
#                 for k in range(t, len(reward_arr)-1):
#                     a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
#                             (1-int(dones_arr[k])) - values[k])
#                     discount *= self.gamma*self.gae_lambda
#                 advantage[t] = a_t
#             advantage = T.tensor(advantage).to(self.actor.device)

#             values = T.tensor(values).to(self.actor.device)
#             for batch in batches:
#                 states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
#                 old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
#                 actions = T.tensor(action_arr[batch]).to(self.actor.device)

#                 dist = self.actor(states)
#                 critic_value = self.critic(states)

#                 critic_value = T.squeeze(critic_value)

#                 new_probs = dist.log_prob(actions)
#                 prob_ratio = new_probs.exp() / old_probs.exp()
#                 #prob_ratio = (new_probs - old_probs).exp()
#                 weighted_probs = advantage[batch] * prob_ratio
#                 weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
#                         1+self.policy_clip)*advantage[batch]
#                 actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

#                 returns = advantage[batch] + values[batch]
#                 critic_loss = (returns-critic_value)**2
#                 critic_loss = critic_loss.mean()

#                 total_loss = actor_loss + 0.5*critic_loss
#                 self.actor.optimizer.zero_grad()
#                 self.critic.optimizer.zero_grad()
#                 total_loss.backward()
#                 self.actor.optimizer.step()
#                 self.critic.optimizer.step()

#         self.memory.clear_memory()

# def plot_learning_curve(x, scores, figure_file):
#     running_avg = np.zeros(len(scores))
#     for i in range(len(running_avg)):
#         running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
#     plt.plot(x, running_avg)
#     plt.title('Running average of previous 100 scores')
#     plt.savefig(figure_file)