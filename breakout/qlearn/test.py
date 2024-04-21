# from ..breakout import Breakout

from breakout import Breakout

from q_learn import Estimator
import numpy as np



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


for est in estimators:

    rewards = []
    for i in range(1):
        state, img, info = env.reset()
        reward_sum = 0
        for i in range(1000):
            action = np.argmax(est.predict(state))
            state, reward, is_done, term, img, info = env.step(action)
            # print(state)
            
            reward_sum += reward
            if is_done or term:
                break
        rewards.append(reward_sum)

    print(sorted(rewards))
