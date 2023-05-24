class ModelConfig:
    def __init__ (self):
        self.model_name              = "GCN"
        self.hidden_channels         = 10
        self.dropout                 = 0.5


class MLPLearnerConfig:
    def __init__ (self):
        self.nlayers=2
        self.isize=150
        self.k=30
        self.knn_metric="cosine_sim"
        self.i=6
        self.sparse=False
        self.act_gl='relu'

class GCLConfig:
    def __init__ (self):
        self.numlayers=2
        self.in_dim=150
        self.hidden_dim= 150 // 2
        self.emb_dim=50
        self.proj_dim=30
        self.dropout_gcl=0.3
        self.dropout_adj=0.2
        self.sparse=None