from breakout import Breakout

from q_learn import Estimator
import numpy as np
import math



env = Breakout()

estimators = []

estimator = Estimator(0, 100, 6, 3, 'cpu')
estimator.load("breakout/qlearn/models/model_24.pth")
estimators.append(estimator)

estimator = Estimator(0, 60, 6, 3, 'cpu')
estimator.load("breakout/qlearn/models/model_20.pth")
estimators.append(estimator)

estimator = Estimator(0, 50, 6, 3, 'cpu')
estimator.load("breakout/qlearn/models/trained_model.pth")
estimators.append(estimator)

n_x = 60

for est in estimators:

    rewards = []
    for i in range(60):
        state, img, info = env.reset()
        reward_sum = 0
        for i in range(10000):
            action = np.argmax(est.predict(state))
            state, reward, is_done, term, img, info = env.step(action)
            # print(state)
            
            reward_sum += reward
            if is_done or term:
                break
        rewards.append(reward_sum)

    print(sorted(rewards))

    a = np.mean(rewards)
    alpha = 0.05
    S = np.std(rewards)
    S_2 = np.var(rewards)
    n_free = n_x - 1
    t_student = 2.000

    print(f'Выборочное среднее: {a}; std: {S}; var {S_2}')
    print(f'Доверительный интервал для выборочного среднего при неизвестной дисперсии: {a - S/math.sqrt(n_free)* t_student}, {a + S / math.sqrt(n_free)* t_student}')
