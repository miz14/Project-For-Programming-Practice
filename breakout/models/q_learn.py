import random
import torch
import math
from torch.autograd import Variable
import torch.nn as nn
import gym
from tqdm import tqdm

# from main import memory, total_reward_episode


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(-1, 42432)
    

def gen_epsilon_greedy_policy(estimator, epsilon, n_action):
    def policy_function(state):
        probs = torch.ones(n_action) * epsilon / n_action
        q_values = estimator.predict(state)
        best_action = torch.argmax(q_values).item()
        probs[best_action] += 1.0 - epsilon
        action = torch.multinomial(probs, 1).item()
        return action
    return policy_function

class Estimator():
    def __init__(self, n_action, n_hidden, lr=0.05):
        # self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        # self.n_feat = n_feat
        self.feature_extractor = feature_extractor()

        self.models = []
        self.optimizers = []

        self.criterion = torch.nn.MSELoss()
        for _ in range(n_action):

            model = nn.Sequential(
                self.feature_extractor,

                nn.Linear(42432, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1)
            )
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)
        
    # def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
    #     """
    #     Генерирует коэффициенты признаков, выбирая их из нормального
    #     распределения
    #     @param n_feat: количество признаков
    #     @param n_state: количество состояний
    #     @param sigma: параметр ядра
    #     @return: коэффициенты признаков
    #     """
    #     torch.manual_seed(0)
    #     w = torch.randn((n_state, n_feat)) * 1.0 / sigma
    #     b = torch.rand(n_feat) * 2.0 * math.pi
    #     return w, b

    # def get_feature(self, s):
    #     """
    #     Генерирует признаки по входному состоянию
    #     @param s: входное состояние
    #     @return: признаки
    #     """
    #     # print(s)
    #     features = (2.0 / self.n_feat) ** .5 * torch.cos(
    #         torch.matmul(torch.tensor(s).float(), self.w) + self.b)
    #     return features

    def update(self, img, a, y):
        """
        Обновляет веса линейной оценки на основе переданного обучающего
        примера
        @param s: состояние
        @param a: действие
        @param y: целевое значение

        """
        img = torch.tensor(img).float()
        # features = Variable(self.get_feature(s))
        y_pred = self.models[a](img)
        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))

        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()

    def predict(self, img):
            """
            Вычисляет значения Q-функции от состояния, применяя
            обученную модель
            @param s: входное состояние
            @return: ценности состояния
            """
            # features = self.get_feature(s)
            img = torch.tensor(img).float()
            with torch.no_grad():
                return torch.tensor([model(img)
                                for model in self.models])


def q_learning(env, estimator, n_episode, replay_size, n_action, memory, gamma=1.0, epsilon=0.1, epsilon_decay=.99):
        """
        Алгоритм Q-обучения с аппроксимацией функций и воспроизведением опыта
        @param env: окружающая среда Gym
        @param estimator: объект класса Estimator
        @param replay_size: сколько примеров использовать при каждом
                  обновлении модели
        @param n_episode: количество эпизодов
        @param gamma: коэффициент обесценивания
Пакетная обработка с применением буфера воспроизведения опыта  193
        @param epsilon: параметр ε-жад­ной стратегии
        @param epsilon_decay: коэффициент затухания epsilon
        """
        total_reward_episode = [0] * n_episode

        for episode in tqdm(range(n_episode)):
            policy = gen_epsilon_greedy_policy(estimator,
                            epsilon * epsilon_decay ** episode,
                            n_action)
            state, info = env.reset()
            is_done = False
            step = 0
            while not is_done and step < 1000:
                step += 1
                action = policy(state)
                next_state, reward, is_done, trancation, info = env.step(action)
                total_reward_episode[episode] += reward
                if is_done:
                    break
                if trancation:
                    break
                q_values_next = estimator.predict(next_state)
                td_target = reward + gamma * torch.max(q_values_next)
                memory.append((state, action, td_target))
                state = next_state
                
            replay_data = random.sample(memory, min(replay_size, len(memory)))
            for state, action, td_target in replay_data:
                estimator.update(state, action, td_target)

        return total_reward_episode