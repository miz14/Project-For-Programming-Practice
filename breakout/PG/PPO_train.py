import cv2
import gym
import numpy as np
from PPO import Agent, plot_learning_curve

class Breakout:
    def __init__(self, resize=(72, 72), stack_size = 4, frame_skip = 10, max_steps = None):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.frame_skip = frame_skip
        self.stack_size = stack_size
        self.stack = None
        self.resize = resize
        self.n_state = (stack_size, resize[0], resize[1])
        self.n_action = self.env.action_space.n - 1
        self.max_steps = max_steps
        self.step_num = 0
        self.lives = None


    def resize_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        return img
    
    def stack_img(self, img, create = False):  
        if create:
            self.stack = [img] * self.stack_size
        else:
            self.stack.pop(0)
            self.stack.append(img)
        return np.stack(self.stack, axis=0)


    def reset(self):
        img, info = self.env.reset()
        img = self.resize_img(img)
        stack = self.stack_img(img, create=True)
        self.step_num = 0
        self.lives = None
        return stack, info

    def step(self, action):
        if action > 0:
            action += 1
        img = None
        reward_sum = 0
        is_done = False
        truncated = False
        info = None
        
        for i in range(self.frame_skip):
            if self.lives is None:
                img, reward, is_done, truncated, info = self.env.step(1)
                self.lives = info['lives']

            img, reward, is_done, truncated, info = self.env.step(action)
            if info['lives'] < self.lives:
                self.lives -= 1
                # reward_sum -= 10
                img, reward, is_done, truncated, info = self.env.step(1)

            action = 0
            # if action == 3:
            #     action = 2
            # elif action == 2:
            #     action = 0
            reward_sum += reward

            if is_done or truncated:
                break
        if not self.max_steps is None:

            self.step_num += 1
            if self.step_num > self.max_steps:
                is_done = True
        img = self.resize_img(img)
        stack = self.stack_img(img)
        return stack, reward_sum, is_done, truncated, info


env = Breakout(resize=(82, 82), frame_skip=3)

N = 20
batch_size = 4
n_epochs = 100
alpha = 0.0003
agent = Agent(n_actions=env.n_action, batch_size=batch_size,
                alpha=alpha, n_epochs=n_epochs,
                n_state = env.n_state,
                img_model_out_state=5776)
n_games = 300

figure_file = 'cartpole.png'

best_score = 10
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_games):
    observation, info = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, t, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)