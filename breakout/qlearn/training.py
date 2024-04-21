from collections import deque
import matplotlib.pyplot as plt
from q_learn import Estimator, q_learning
from breakout import Breakout


memory = deque(maxlen=100)

env = Breakout()
state, img, info = env.reset()

n_action = env.action_space

n_feature = 100
n_hidden = 20
lr = 1e-4
estimator = Estimator(n_feature, n_hidden, n_state=6, n_action=3, device='cpu', lr=lr)

n_episode = 1000
replay_size = 90
total_reward_episode = q_learning(env, estimator, n_episode, replay_size, n_action, memory, gamma=1, epsilon=0.1)

estimator.save("breakout/models/saves/qlearn/trained1_model.pth")

plt.plot(total_reward_episode, 'b.')
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()