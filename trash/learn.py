import gym
from matplotlib import pyplot as plt
from trash.encoder import ImageEncoder
from trash.estimator import Estimator
from trash.q_learning import q_learning
import torch.nn as nn
import torch

model = ImageEncoder()
model.load_state_dict(torch.load('image_encoder_model.pth'))
# model.load_state_dict(torch.load('image_encoder_model.pth'))

criterion = nn.CrossEntropyLoss()  # Пример функции потерь для классификации
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
env = gym.make("ALE/Breakout-v5")
state, info = env.reset()
train_loader = []
train_loader.append(state)
for i in range(200):
    state, r, t, t, info = env.step(env.action_space.sample())
    train_loader.append(state)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        optimizer.step()
 
    print(f"Epoch [{epoch+1}/{num_epochs}]")


model.eval()

torch.save(model.state_dict(), 'image_encoder_model.pth')

n_state = 32
n_action = env.action_space.n
n_feature = 200
n_hidden = 50
lr = 0.001
n_episode = 100

estimator = Estimator(n_feature, n_state, n_action, n_hidden, lr)
total_reward_episode = q_learning(env, model, estimator, n_episode, n_action, epsilon=0.4)
print(total_reward_episode)
estimator.save('trained_model.pth')

plt.plot(total_reward_episode)
plt.title('Зависимсоть вознаграждения')
plt.xlabel('Эпизод')
plt.ylabel('Полное вознаграждение')
plt.show()