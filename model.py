import torch
import torch.nn as nn


class ParetoSetModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj, problem):
        super(ParetoSetModel, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.problem = problem
        self.fc1 = nn.Linear(self.n_obj, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_dim)
        self.xl = torch.tensor(self.problem.xl)
        self.xu = torch.tensor(self.problem.xu)
    def forward(self, pref):
        if pref.device.type == 'cuda':
            self.lbound = self.xl.cuda()
            self.ubound = self.xu.cuda()

        x = torch.relu(self.fc1(pref))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        x = torch.sigmoid(x)
        x = x * (self.ubound - self.lbound) + self.lbound
        return x.to(torch.float64)