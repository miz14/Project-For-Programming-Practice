import cv2
import pygame
from breakout import Breakout
import matplotlib.backends.backend_agg as agg
import matplotlib
import pylab
import matplotlib.pyplot as plt
matplotlib.use("Agg")

pygame.init()
screen = pygame.display.set_mode(size=(640, 480))
pygame.display.set_caption("Умный муравей")
clock = pygame.time.Clock()

FPS = 30

env = Breakout()

observation, info = env.reset()
env.rec()

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

lives = 5
while running:
    clock.tick(FPS)
    
    surf = get_image(observation)
    screen.blit(surf, dest = (0,0))
    action = env.get_best_action(observation)
    pre_obs = observation
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
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
env.stop_rec()