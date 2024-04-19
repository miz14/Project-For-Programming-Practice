import math
from torch.autograd import Variable
import torch
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
        torch.manual_seed(0)
        w = torch.randn((n_state, n_feat)) * 1.0 / sigma
        b = torch.rand(n_feat) * 2.0 * math.pi
        return w, b
    
    def get_feature(self, s):
        features = (2.0 / self.n_feat) ** 0.5 * torch.cos(torch.matmul(torch.tensor(s).float(), self.w) + self.b)
        return features

    def update(self, s, a, y, is_dead):
        features = Variable(self.get_feature(s))
        y_pred = self.models[a](features)

        if is_dead:
            loss = self.criterion(y_pred, Variable(torch.tensor([y]))) + 500
        else:
            loss = self.criterion(y_pred, Variable(torch.tensor([y])))
        self.optimizers[a].zero_grad()
        loss.backward()
        self.optimizers[a].step()

    def predict(self, s):
        features = Variable(self.get_feature(s))
        with torch.no_grad():
            probabilities = torch.tensor([model(features) for model in self.models])
            return probabilities
        
    def save(self, path):
        state_dict = {'models': [model.state_dict() for model in self.models],
                        'optimizers': [optimizer.state_dict() for optimizer in self.optimizers]}
        torch.save(state_dict, path)

    def load(self, path):
        state_dict = torch.load(path)
        for model, optimizer, saved_model_state, saved_optimizer_state in zip(self.models, self.optimizers, state_dict['models'], state_dict['optimizers']):
            model.load_state_dict(saved_model_state)
            optimizer.load_state_dict(saved_optimizer_state)