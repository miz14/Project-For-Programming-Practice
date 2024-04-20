# import numpy as np
# import gym
# import matplotlib.pyplot as plt
# import torch.nn as nn
# # from models.main import Breakout
# import cv2
# import random
# # rgb
# # frameskip=(2, 5) - пока хз, что это
# # repeat_action_probability = 0 - вероянтось повторения действий


# # Кирпичи
# # (57, 8)
# #         (93, 152)

# # Размер кирпича (6, 8)

# # Шар
# # (93, 8)
# #         (189, 152)

# # Доска
# # (189, 8)
# #         (193, 152)



# class Breakout:

#     def __init__(self):
#         self.env = gym.make('Breakout-v4', obs_type="grayscale")
#         self.size = None
#         self.next_state = None
#         self.action_space = 3 # 1 - сгенерировать шар

#         self.bricks_coords = {'x': (57, 93), 'y': (8, 152)}
#         self.brick_size = (6, 8)
#         self.ball_coords = {'x': (93, 189), 'y': (8, 152)}
#         self.board_coords = {'x': (189, 193), 'y': (8, 152)}
#         self.lives = None

#     def get_features(self, img):
#         features_vector = []

#         # Создаем маску изображения из 0 и 1
#         _, mask_img = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)


#         # Проверям наличие кирпичей (144 штук)
#         for i in range(self.bricks_coords['x'][0], self.bricks_coords['x'][1], self.brick_size[0]):
#             for j in range(self.bricks_coords['y'][0], self.board_coords['y'][1], self.brick_size[1]):
#                 if np.all(mask_img[i: i + self.brick_size[0], j: j + self.brick_size[1]]):
#                     features_vector.append(1)
#                 else:
#                     features_vector.append(0)

#         # Поиск координат шара
#         contours, _ = cv2.findContours(mask_img[self.ball_coords['x'][0] : self.ball_coords['x'][1], self.ball_coords['y'][0] : self.ball_coords['y'][1]],
#                                        cv2.RETR_TREE,
#                                        cv2.CHAIN_APPROX_SIMPLE)
        
#         # Если нет шара
#         if len(contours) == 0:
#             # -1, 72
#             features_vector.append(-1)
#             features_vector.append(72)
        
#         else:

#             left_top = contours[0][0][0]
#             left_bottom = contours[0][1][0]

#             # Если шар виден не полностью
#             if left_bottom[1] - left_top[1] + 1 < 4:

#                 # Не виден внизу
#                 if left_top[1] > 50:
#                     features_vector.append(left_top[1] + 3)
#                     features_vector.append(left_bottom[0])
#                 # Не вден вверху
#                 else:
#                     features_vector.append(left_bottom[1])
#                     features_vector.append(left_bottom[0])

#             # Виден полностью
#             else:
#                 features_vector.append(left_bottom[1])
#                 features_vector.append(left_bottom[0])

#         # Поиск координат доски
#         contours, _ = cv2.findContours(mask_img[self.board_coords['x'][0] : self.board_coords['x'][1], self.board_coords['y'][0] : self.board_coords['y'][1]],
#                                     cv2.RETR_TREE,
#                                     cv2.CHAIN_APPROX_SIMPLE)
        
#         # Если в зоне доски 2 элемента
#         contour = None
#         if len(contours) == 2:
#             if len(contours[0]) == 2:
#                 contour = contours[1]
#             elif len(contours[1]) == 2:
#                 contour = contours[0]
#             elif contours[0][3][0][0] -  contours[0][0][0][0] < 4:
#                 contour = contours[1]
#             else:
#                 contour = contours[0]
#         else:
#             contour = contours[0]

#         left_top = contour[0][0]
#         right_top = contour[3][0]

#         # Если доска видна не полностью
#         if right_top[0] - left_top[0] + 1 < 16:
#             # Видна не полностью слева
#             if left_top[0] < 70:
#                 features_vector.append(right_top[0] - 16)

#             # Видна не полностью справа
#             else:
#                 features_vector.append(left_top[0])
#         # Видна полностю
#         else:
#             features_vector.append(left_top[0])
        
#         return features_vector
 
#     def reset(self):
#         img, info = self.env.reset()
#         self.lives = info['lives']
#         state = self.get_features(img)
#         return state, img, info
  
#     def step(self, action):
#         if action > 0:
#             action += 1
#         if self.lives is None:
#             next_img, reward, is_done, trancation, info =  self.env.step(1)
#             self.lives = info['lives']

#         next_img, reward, is_done, trancation, info =  self.env.step(action)
#         if info['lives'] < self.lives:
#             next_img, reward, is_done, trancation, info =  self.env.step(1)
#             reward -= 20
#         next_state = self.get_features(next_img)
#         return next_state, reward, is_done, trancation, next_img, info
    

# env = Breakout()
# print(env.action_space)

# # state, img, info = env.reset()

# # state, reward, is_done, trancation, img, info = env.step(1)
# # for i in range(10000):
# #     state, reward, is_done, trancation, img, info = env.step(2)
# #     if reward == -20:
        
# #         print(len(state), state)
# #         plt.imshow(img, cmap='gray') #interpolation='nearest', aspect='auto'
# #         plt.show()

