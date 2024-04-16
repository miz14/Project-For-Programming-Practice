import time
import gym

import pygame

import matplotlib.backends.backend_agg as agg

import numpy as np

import matplotlib

import pylab

import matplotlib.pyplot as plt

matplotlib.use("Agg")

pygame.init()
screen = pygame.display.set_mode(size=(640, 480))
pygame.display.set_caption("Умный муравей")
clock = pygame.time.Clock()



FPS = 30

env = gym.make("ALE/Breakout-v5")

observation, info = env.reset()

def get_image(observ):
    fig = pylab.figure()
    ax = fig.gca()
    ax.imshow(observ)
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    print(size)
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    return surf

running = True
while running:

    clock.tick(FPS)
    surf = get_image(observation)
    screen.blit(surf, dest = (0,0))
    observation,reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated:
        observation, info = env.reset()
    pressed = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    pygame.display.flip()