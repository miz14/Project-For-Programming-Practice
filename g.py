import time
import gym

import pygame

import matplotlib.backends.backend_agg as agg

import numpy as np

import matplotlib

import pylab

import matplotlib.pyplot as plt

matplotlib.use("Agg")

fig = pylab.figure()

ax = fig.gca()

WIDTH = 800

HEIGHT = 800




pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Умный муравей")
clock = pygame.time.Clock()



FPS = 10

env = gym.make("ALE/Breakout-v5")

observation, info = env.reset()

running = True
while running:

    clock.tick(FPS)

    pressed = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False


    for _ in range(500):
        observation,reward, terminated, truncated, info = env.step(env.action_space.sample()) 
        ax.imshow(observation)
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, (0,0))
        pygame.display.flip()