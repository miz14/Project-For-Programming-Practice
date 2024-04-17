import random
import torch
import math
from torch.autograd import Variable
# from main import memory, total_reward_episode

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
    def __init__(self, n_feat, n_state, n_action, n_hidden, lr=0.05):
        self.w, self.b = self.get_gaussian_wb(n_feat, n_state)
        self.n_feat = n_feat
        self.models = []
        self.optimizers = []
        self.criterion = torch.nn.MSELoss()
        for _ in range(n_action):
            model = torch.nn.Sequential(
                        torch.nn.Linear(n_feat, n_hidden),
                        torch.nn.ReLU(),
                        torch.nn.Linear(n_hidden, 1)
            )
            self.models.append(model)
            optimizer = torch.optim.Adam(model.parameters(), lr)
            self.optimizers.append(optimizer)
        
    def get_gaussian_wb(self, n_feat, n_state, sigma=.2):
        """
        Генерирует коэффициенты признаков, выбирая их из нормального
        распределения
        @param n_feat: количество признаков
        @param n_state: количество состояний
        @param sigma: параметр ядра
        @return: коэффициенты признаков
        """
        torch.manual_seed(0)
        w = torch.randn((n_state, n_feat)) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * math.pi
        return w, b

    def get_feature(self, s):
        """
        Генерирует признаки по входному состоянию
        @param s: входное состояние
        @return: признаки
        """
        # print(s)
        features = (2.0 / self.n_feat) ** .5 * torch.cos(
            torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features

    def update(self, s, a, y):
        """
        Обновляет веса линейной оценки на основе переданного обучающего
        примера
        @param s: состояние
        @param a: действие
        @param y: целевое значение
        """
        features = Variable(self.get_feature(s))
        y_pred = self.models[a](features)
        loss = self.criterion(y_pred, Variable(torch.Tensor([y])))
        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()
    def predict(self, s):
            """
            Вычисляет значения Q-функции от состояния, применяя
            обученную модель
            @param s: входное состояние
            @return: ценности состояния
            """
            features = self.get_feature(s)
            with torch.no_grad():
                return torch.tensor([model(features)
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

        for episode in range(n_episode):
            policy = gen_epsilon_greedy_policy(estimator,
                            epsilon * epsilon_decay ** episode,
                            n_action)
            state, info = env.reset()
            is_done = False
            while not is_done:
                action = policy(state)
                next_state, reward, is_done, trancation, _ = env.step(action)
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