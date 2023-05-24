class TrainConfig:
    def __init__ (self):
        self.batch_size              = 30
        self.epoch                   = 200
        # self.lr                      = 0.0009
        self.lr                      = 0.003
        # self.weight_decay            = 0.0005
        self.weight_decay            = None
        self.device                  = "cuda:0"
        self.verbose                 = False


class TrainConfigGSL:
    def __init__ (self):
        self.epoches                 = 200
        self.c                       = 0
        self.tau                     = 0.9999
        self.batch_size              = 20
        self.lr_proj                 = 0.01
        self.lr_gl                   = 0.01
        self.lr_gnn                  = 0.003
        self.w_decay                 = 0.0
        self.eval_freq               = 5
        self.verbose                 = True
        self.device                  = "cuda:0"
        self.weight_decay_eval       = None