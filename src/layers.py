from abc import ABC

import torch
import torch.nn as nn
import numpy as np


class FeaturesLinear(nn.Module, ABC):
    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array([0, *np.cumsum(field_dims)[:-1]], dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(nn.Module, ABC):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array([0, *np.cumsum(field_dims)[:-1]], dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)


class FactorizationMachine(nn.Module, ABC):
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        logit = square_of_sum - sum_of_square

        if self.reduce_sum:
            logit = torch.sum(logit, dim=1, keepdim=True)

        return 0.5 * logit


class MultiLayerPerceptron(nn.Module, ABC):
    def __init__(self, input_dim, embed_dims, dropout, is_bn=True, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()

        layers = []

        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            if is_bn:
                layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

