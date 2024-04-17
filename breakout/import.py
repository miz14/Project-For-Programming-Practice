import numpy as np
import gym
import matplotlib.pyplot as plt

# rgb
# frameskip=(2, 5) - пока хз, что это
# repeat_action_probability = 0 - вероянтось повторения действий

env = gym.make('Breakout-v4')
env.reset()
save = None
for i in range(50):
    action = env.action_space.sample()
    save = env.step(action)
print(save)
img = env.step(action)[0]
plt.imshow(img, interpolation='nearest', aspect='auto')
# plt.imshow(img)
plt.show()

