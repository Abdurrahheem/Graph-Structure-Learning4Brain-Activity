import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tnn
from config.config import Config
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv

torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

class MODEL(torch.nn.Module):
    def params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

class GCN(MODEL):
    def __init__(self, cfg):
        super(GCN, self).__init__()

        if cfg.use_node_embeddings:
            self.conv1 = tnn.GCNConv(cfg.feature_size, cfg.hidden_channels)
        else:
            self.conv1 = tnn.GCNConv(cfg.N_rois, cfg.hidden_channels)

        self.lin = nn.Linear(cfg.hidden_channels, cfg.classes)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.bn = nn.BatchNorm1d(cfg.hidden_channels)
        # self.act = nn.Tanh()

    def forward(self, x, edge_index, batch, edge_weights=None):

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index=edge_index, edge_weight=edge_weights)
        x = self.dropout(x)
        x = self.act(x)
        # x = self.bn(x)

        # 2. Readout layer
        x_pool = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x_pool)
        return x, x_pool

    def compute_l1_loss(self, w):
          return torch.abs(w).sum()