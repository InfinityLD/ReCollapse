from abc import ABC

import torch
import torch.nn as nn

from src.layers import FeaturesLinear, FeaturesEmbedding, FactorizationMachine


class FactorizationMachineModel(nn.Module, ABC):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims=field_dims)
        self.embedding = FeaturesEmbedding(field_dims=field_dims, embed_dim=embed_dim)
        self.fm = FactorizationMachine()

    def forward(self, x):
        logit = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(logit.squeeze(1))
