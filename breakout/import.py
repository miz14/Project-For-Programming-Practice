import numpy as np
import gym
import matplotlib.pyplot as plt
import torch.nn as nn
# rgb
# frameskip=(2, 5) - пока хз, что это
# repeat_action_probability = 0 - вероянтось повторения действий

class Env:

    def __init__(self):
        self.env = gym.make('Breakout-v4')
        self.size = None
        self.next_state = None
        #144 164
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2), #16, 72, 82
            nn.Conv2d(16, 32, kernel_size=3, stride=2), #32 36 41
        )
            x.view(-1, 32*36*41)


    def get_next_state():
        pass

    
    def reset(self):

        #8, 31,  ->  151, 194  - 144 164
        
        img, info = self.env.reset()
        self.size = img.shape

    
    def step():
        pass



env = gym.make('Breakout-v4', obs_type="grayscale")
save = env.reset()
print(save)
save = None
for i in range(50):
    action = env.action_space.sample()
    save = env.step(action)
print(save)
img = env.step(action)[0]

plt.imshow(img, interpolation='nearest', aspect='auto', cmap='gray')
# plt.imshow(img)
plt.show()

