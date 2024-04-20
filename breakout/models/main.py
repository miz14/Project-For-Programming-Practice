from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt
from q_learn import Estimator, q_learning
import cv2

class Breakout:

    def __init__(self):
        self.env = gym.make('Breakout-v4', obs_type="grayscale")
        self.size = None
        self.next_state = None
        self.action_space = 3 # 1 - сгенерировать шар

        self.bricks_coords = {'x': (57, 93), 'y': (8, 152)}
        self.brick_size = (6, 8)
        self.ball_coords = {'x': (93, 189), 'y': (8, 152)}
        self.board_coords = {'x': (189, 193), 'y': (8, 152)}
        self.lives = None

    def get_features(self, img):
        features_vector = []

        # Создаем маску изображения из 0 и 1
        _, mask_img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)


        # # Проверям наличие кирпичей (144 штук)
        # for i in range(self.bricks_coords['x'][0], self.bricks_coords['x'][1], self.brick_size[0]):
        #     for j in range(self.bricks_coords['y'][0], self.board_coords['y'][1], self.brick_size[1]):
        #         if np.all(mask_img[i: i + self.brick_size[0], j: j + self.brick_size[1]]):
        #             features_vector.append(1)
        #         else:
        #             features_vector.append(0)

        # Поиск координат шара
        contours, _ = cv2.findContours(mask_img[self.ball_coords['x'][0] : self.ball_coords['x'][1], self.ball_coords['y'][0] : self.ball_coords['y'][1]],
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        # Если нет шара
        if len(contours) == 0:
            # -1, 72
            features_vector.append(72)
            features_vector.append(-1)
        
        else:

            x, y, width, height = cv2.boundingRect(contours[0])

            # Если шар виден не полностью
            if height < 4:


                # Не виден внизу
                if y > 50:
                    features_vector.append(x)
                    features_vector.append(y + height - 1)
                # Не вден вверху
                else:
                    features_vector.append(x)
                    features_vector.append(y)

            # Виден полностью
            else:
                features_vector.append(x)
                features_vector.append(y)

        # Поиск координат доски
        contours, _ = cv2.findContours(mask_img[self.board_coords['x'][0] : self.board_coords['x'][1], self.board_coords['y'][0] : self.board_coords['y'][1]],
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Если в зоне доски 2 элемента
        contour = None
        if len(contours) == 2:
            if len(contours[0]) < 4:
                contour = contours[1]
            elif len(contours[1]) < 4:
                contour = contours[0]

            else:
                t_x, t_y, t_widht, t_height = cv2.boundingRect(contours[0])
                if t_widht < 4:
                    contour = contours[1]
                else:
                    contour = contours[0]
        else:
            contour = contours[0]

        x, y, width, height = cv2.boundingRect(contour)

        # Если доска видна не полностью
        if width < 16:
            # Видна не полностью слева
            if x < 70:
                features_vector.append(x - 16 + width)

            # Видна не полностью справа
            else:
                features_vector.append(x)
        # Видна полностю
        else:
            features_vector.append(x)
        
        return features_vector
 
    def reset(self):
        img, info = self.env.reset()
        next_img, reward, is_done, trancation, info =  self.env.step(1)
        self.lives = info['lives']
        state = self.get_features(next_img)
        return state, next_img, info
  
    def step(self, action):
        if action > 0:
            action += 1
        next_img, reward, is_done, trancation, info =  self.env.step(action)
        if info['lives'] < self.lives:
            next_img, reward, is_done, trancation, info =  self.env.step(1)
            self.lives -= 1
            # reward -= 10
        next_state = self.get_features(next_img)
        return next_state, reward, is_done, trancation, next_img, info

memory = deque(maxlen=1000)

env = Breakout()
state, img, info = env.reset()

n_action = env.action_space

n_feature = 100
n_hidden = 40
lr = 1e-4
estimator = Estimator(n_feature, n_hidden, n_state=3, n_action=3, device='cpu', lr=lr)

n_episode = 1000
replay_size = 500
total_reward_episode = q_learning(env, estimator, n_episode, replay_size, n_action, memory, gamma=0.8, epsilon=0.1)

estimator.save("breakout/models/saves/qlearn/trained_model.pth")

plt.plot(total_reward_episode, 'b.')
plt.title('Зависимсоть вознаграждения в эпизоде от времени')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()