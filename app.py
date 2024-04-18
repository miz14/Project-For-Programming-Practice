import gym

import pygame

import matplotlib.backends.backend_agg as agg

import matplotlib

import pylab

import matplotlib.pyplot as plt
import torch

from encoder import ImageEncoder
from estimator import Estimator
from q_learning import q_learning


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
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    plt.close()
    return surf

running = True
num_actions = env.action_space.n

env.step(1)
lives = 5

encoder = ImageEncoder()

n_state = 32
n_action = env.action_space.n
n_feature = 200
n_hidden = 50
lr = 0.001

estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)

estimator.load('trained_model.pth')

while running:
    clock.tick(FPS)
    surf = get_image(observation)
    screen.blit(surf, dest = (0,0))

    input_image = observation 
    output_value = encoder(input_image)
    observation = output_value

    q_values = estimator.predict(observation)
    action = torch.argmax(q_values)

    observation, reward, terminated, truncated, info = env.step(action.item())

    if info["lives"] < lives:
        lives -= 1
        observation, reward, terminated, truncated, info = env.step(1)
    if terminated:
        lives = 5
        observation, info = env.reset()
        observation, reward, terminated, truncated, info = env.step(1)
    pressed = pygame.key.get_pressed()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    pygame.display.flip()