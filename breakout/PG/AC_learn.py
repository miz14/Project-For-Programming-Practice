import cv2
from matplotlib import pyplot as plt
from ActorCritic import reinforce, PolicyNetwork
import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import numpy as np
import torch
from collections import deque
import random

torch.set_default_device('cuda')

# env = Breakout()

class Breakout:
    def __init__(self, resize=(72, 72), stack_size = 4, frame_skip = 10, max_steps = None):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.stack = None
        self.resize = resize
        self.n_state = (stack_size, resize[0], resize[1])
        self.n_action = self.env.action_space.n - 1
        self.max_steps = max_steps
        self.step_num = 0
        self.lives = None


    def resize_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        return img
    
    def stack_img(self, img, create = False):  
        if create:
            self.stack = [img] * self.stack_size
        else:
            self.stack.pop(0)
            self.stack.append(img)
        return np.stack(self.stack, axis=0)


    def reset(self):
        img, info = self.env.reset()
        img = self.resize_img(img)
        stack = self.stack_img(img, create=True)
        self.step_num = 0
        self.lives = None
        return stack, info

    def step(self, action):
        if action > 0:
            action += 1
        img = None
        reward_sum = 0
        is_done = False
        truncated = False
        info = None
        
        for i in range(self.frame_skip):
            if self.lives is None:
                img, reward, is_done, truncated, info = self.env.step(1)
                self.lives = info['lives']

            img, reward, is_done, truncated, info = self.env.step(action)
            if info['lives'] < self.lives:
                self.lives -= 1
                # reward_sum -= 10
                img, reward, is_done, truncated, info = self.env.step(1)

            action = 0
            # if action == 3:
            #     action = 2
            # elif action == 2:
            #     action = 0
            reward_sum += reward

            if is_done or truncated:
                break
        if not self.max_steps is None:

            self.step_num += 1
            if self.step_num > self.max_steps:
                is_done = True
        img = self.resize_img(img)
        stack = self.stack_img(img)
        return stack, reward_sum, is_done, truncated, info




# env = Breakout(resize=(84, 84), frame_skip=3)

env = FrameStack(AtariPreprocessing(gym.make('BreakoutNoFrameskip-v4'), screen_size=84), 4)

# fig, ax = plt.subplots(1, 2, dpi=150)

# i1, _ = env1.reset()
# i2, _ = env2.reset(seed=42)
# ax[0].imshow(i1[3], cmap='gray')
# ax[1].imshow(i2[3], cmap='gray')
# plt.show()


# for i in range(100):
#     # e = random.randint(2, 3)
#     e = 2
    
#     print(e)
#     i1, _, _, _, info1 = env1.step(2)
#     i2, _, _, _, info2 = env2.step(3)
#     print(info1['frame_number'], info2['frame_number'])
#     fig, ax = plt.subplots(1, 2, dpi=150)
#     ax[0].imshow(i1[3], cmap='gray')
#     ax[1].imshow(i2[3], cmap='gray')
#     plt.show()
# err

# env = Breakout()



# img, _ = env.reset()
# # plt.imshow(img[0], cmap="gray")
# # plt.show()
# # plt.imshow(img[1], cmap="gray")
# # plt.show()
# # plt.imshow(img[2], cmap="gray")
# # plt.show()
# env.step(1)
# plt.imshow(img[3], cmap="gray")
# plt.show()
# env.step(1)
# for i in range(20):
#     img, reward, is_done, t, info = env.step(random.randint(0, 3))
#     print(reward, is_done, t, info)
#     # plt.imshow(img[0], cmap="gray")
#     # plt.show()
#     # plt.imshow(img[1], cmap="gray")
#     # plt.show()
#     # plt.imshow(img[2], cmap="gray")
#     # plt.show()
#     plt.imshow(img[3], cmap="gray")
#     plt.show()


# n_state = env.n_state
# n_action = env.n_action

n_state = env.observation_space.shape
n_action = env.action_space.n 


# n_hidden = [32, 16]
# lr = 0.1

# n_episode = 1000
# gamma = 0.9

# policy_net = PolicyNetwork(n_state, n_action, n_hidden, lr)

# total_reward_episode = re(env, policy_net, n_episode, gamma)

lr = 1e-4
policy_net = PolicyNetwork(n_state, n_action, lr)

gamma = 0.9

n_episode = 1000
total_reward_episode = reinforce(env, policy_net, n_episode, gamma)


plt.plot(total_reward_episode, '.')
plt.title("Episode reward over time")
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()

# N = 20
# batch_size = 5
# n_epochs = 4
# alpha = 0.0003
# agent = Agent(n_actions=n_action, batch_size=batch_size,
#                 alpha=alpha, n_epochs=n_epochs,
#                 input_dims=[n_state])
# n_games = 300

# figure_file = 'cartpole.png'

# best_score = 0
# score_history = []

# learn_iters = 0
# avg_score = 0
# n_steps = 0

# for i in range(n_games):
#     observation, _, info = env.reset()
#     done = False
#     score = 0
#     while not done:
#         action, prob, val = agent.choose_action(observation)
#         observation_, reward, done, t, _, info = env.step(action)
#         n_steps += 1
#         score += reward
#         agent.remember(observation, action, prob, val, reward, done)
#         if n_steps % N == 0:
#             agent.learn()
#             learn_iters += 1
#         observation = observation_
#     score_history.append(score)
#     avg_score = np.mean(score_history[-100:])

#     if avg_score > best_score:
#         best_score = avg_score
#         # agent.save_models()

#     print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
#             'time_steps', n_steps, 'learning_steps', learn_iters)
# x = [i+1 for i in range(len(score_history))]
# plot_learning_curve(x, score_history, figure_file)
