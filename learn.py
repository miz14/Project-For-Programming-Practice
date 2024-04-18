import gym
from matplotlib import pyplot as plt
from estimator import Estimator
from q_learning import q_learning



env = gym.make("ALE/Breakout-v5")

n_state = 32
n_action = env.action_space.n
n_feature = 200
n_hidden = 50
lr = 0.001
n_episode = 1000

estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)
estimator.load('trained_model.pth')
total_reward_episode = q_learning(env, estimator, n_episode, n_action, epsilon=0.4)
print(total_reward_episode)
estimator.save('trained_model.pth')

plt.plot(total_reward_episode)
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()