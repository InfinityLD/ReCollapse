from abc import ABC

import torch
import torch.nn as nn

from src.layers import FeaturesLinear, FeaturesEmbedding, FactorizationMachine, MultiLayerPerceptron


class DeepFMModel(nn.Module, ABC):
    def __init__(self, field_dims, embed_dim):
        super(DeepFMModel, self).__init__()
        self.linear = FeaturesLinear(field_dims=field_dims)
        self.embedding = FeaturesEmbedding(field_dims=field_dims, embed_dim=embed_dim)
        self.fm = FactorizationMachine()
        self.dnn = MultiLayerPerceptron(input_dim=len(field_dims)*embed_dim, embed_dims=[256, 128], dropout=0.8)

    def forward(self, x):
        embedding_vectors = self.embedding(x)
        linear_logit = self.linear(x)
        fm_logit = torch.sigmoid(self.fm(embedding_vectors))
        dense_embedding_vectors = nn.Flatten()(embedding_vectors)
        dnn_logit = self.dnn(dense_embedding_vectors)
        logit = linear_logit + fm_logit + dnn_logit
        return torch.sigmoid(logit.squeeze(1))

