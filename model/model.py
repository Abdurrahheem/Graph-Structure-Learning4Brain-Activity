import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as tnn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
# from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GINConv

torch.manual_seed(12345)
class MODEL(torch.nn.Module):
    def params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

class GCN(MODEL):
    def __init__(self, hidden_channels, N_rois, output_size):
        super(GCN, self).__init__()

        self.conv1 = tnn.GCNConv(N_rois, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_size)

    def forward(self, x, edge_index, batch):

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index=edge_index)
        x = x.relu()

        # 2. Readout layer
        x_pool = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x_pool)
        return x, x_pool

    def compute_l1_loss(self, w):
          return torch.abs(w).sum()