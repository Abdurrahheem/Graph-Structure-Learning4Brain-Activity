class TrainConfig:
    def __init__ (self):
        self.batch_size              = 30
        self.epoch                   = 200
        # self.lr                      = 0.0009
        self.lr                      = 0.003
        # self.weight_decay            = 0.0005
        self.weight_decay            = None
        self.device                  = "cuda:0"
        self.verbose                 = True