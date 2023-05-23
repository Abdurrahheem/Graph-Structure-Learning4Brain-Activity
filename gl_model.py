import torch
import torch.nn as nn
import torch.nn.functional as F
from gstl_utils import knn_fast, apply_non_linearity, cal_similarity_graph, top_k

class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        # if self.sparse:
        #     embeddings = self.internal_forward(features)
        #     rows, cols, values = knn_fast(embeddings, self.k, 1000)
        #     rows_ = torch.cat((rows, cols))
        #     cols_ = torch.cat((cols, rows))
        #     values_ = torch.cat((values, values))
        #     values_ = apply_non_linearity(values_, self.non_linearity, self.i)
        #     adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
        #     adj.edata['w'] = values_
        #     return adj
        # else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities