import numpy as np
import gym

# rgb
# frameskip=(2, 5) - пока хз, что это
# repeat_action_probability = 0 - вероянтось повторения действий

env = gym.make('Breakout-v4')
env.reset()
action = env.action_space.sample()
print(env.step(action))

