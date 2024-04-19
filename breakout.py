import cv2
import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo
class Breakout:

    def __init__(self):
        self.env = gym.make('Breakout-v4', render_mode='rgb_array_list')
        self.size = None
        self.next_state = None
        self.action_space = self.env.action_space.n
        #144 164
        
    def sample(self):
        return self.env.action_space.sample()

    def cut_image(self, img):
        img_cut = img[31: 195, 8: 152]
        return img_cut
 
    def reset(self):
        #8, 31,  ->  151, 194  - 144 164
        img, info = self.env.reset()
        # img = self.cut_image(img)
        return img, info
  
    def step(self, action):
        next_img, reward, is_done, trancation, info =  self.env.step(action)
        # next_img = self.cut_image(next_img)
        return next_img, reward, is_done, trancation, info
    
    def detect_rectangle(self, image, obj):
        image = self.cut_image(image)
        if obj == 'ball':
            image = image[62:158, :]
        elif obj == 'board':
            image = image[156:163, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangle_coords = (None, None)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            rectangle_coords = (x + width // 2, y + height // 2)
            break
        
        return rectangle_coords
    
    def get_best_action(self, image):
        x_ball, _ = self.detect_rectangle(image, 'ball')
        x_board, _ = self.detect_rectangle(image, 'board')
        print("ball", x_ball)
        print("board", x_board)
        # 144 164
        if x_ball == None:
            return 1
        elif x_board == None:
            return 0
        if x_ball - x_board < 0.001:
            return 3
        elif (x_ball - x_board) > 0.001:
            return 2
        else:
            return 0
        
    def rec(self):
        self.env = RecordVideo(env=self.env, video_folder='video', episode_trigger=lambda x: x % 3 == 0)
        self.env.start_video_recorder()

    def render(self):
        self.env.render()

    def stop_rec(self):
        self.env.close_video_recorder()