import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
from models.main import Breakout
# rgb
# frameskip=(2, 5) - пока хз, что это
# repeat_action_probability = 0 - вероянтось повторения действий


e = Breakout()

env = gym.make('Breakout-v4', obs_type="grayscale")
save = env.reset()

save = None
for i in range(50):
    action = env.action_space.sample()
    save = env.step(action)

img = env.step(action)[0]
img = e.cut_image(img)

plt.imshow(img, interpolation='nearest', aspect='auto', cmap='gray')
# plt.imshow(img)
plt.show()

