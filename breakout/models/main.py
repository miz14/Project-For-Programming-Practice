from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learn import Estimator, q_learning

class Breakout:

    def __init__(self):
        self.env = gym.make('Breakout-v4', obs_type="grayscale")
        self.size = None
        self.next_state = None
        self.action_space = self.env.action_space.n
        #144 164
        
    def cut_image(self, img):
        img_cut = np.expand_dims(img[31: 195, 8: 152], axis=0)
        return img_cut
 
    def reset(self):
        #8, 31,  ->  151, 194  - 144 164
        img, info = self.env.reset()
        img = self.cut_image(img)
        return img, info
  
    def step(self, action):
        next_img, reward, is_done, trancation, info =  self.env.step(action)
        next_img = self.cut_image(next_img)
        return next_img, reward, is_done, trancation, info

memory = deque(maxlen=400)

env = Breakout()

n_action = env.action_space

# n_feature = 200
n_hidden = 50
lr = 0.001
estimator = Estimator(n_action, n_hidden, lr)

n_episode = 10
replay_size = 200
total_reward_episode = q_learning(env, estimator, n_episode, replay_size, n_action, memory, epsilon=0.1)

plt.plot(total_reward_episode, 'b.')
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()