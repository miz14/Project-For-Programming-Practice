from collections import deque
import gym
import matplotlib.pyplot as plt
from q_learn import Estimator, q_learning

memory = deque(maxlen=400)

env = gym.make('CartPole-v0')

n_state = env.observation_space.shape[0]
n_action = env.action_space.n
print(n_state)
n_feature = 200
n_hidden = 50
lr = 0.001
estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)

n_episode = 10
replay_size = 200
total_reward_episode = q_learning(env, estimator, n_episode, replay_size, n_action, memory, epsilon=0.1)

plt.plot(total_reward_episode, 'b.')
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()