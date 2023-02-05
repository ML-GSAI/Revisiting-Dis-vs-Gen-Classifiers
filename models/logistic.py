import torch
from torch import nn

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(features, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, val=0)

class MulticlassLogisticRegressionModel(torch.nn.Module):
    def __init__(self, features,K):
        super(MulticlassLogisticRegressionModel, self).__init__()
        self.features = features
        self.K = K
        self.linear = torch.nn.Linear(features, K)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, val=0)