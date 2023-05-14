class TrainConfig:
    def __init__ (self):
        self.batch_size              = 40
        self.epoch                   = 200
        self.lr                      = 0.0009
        self.weight_decay            = 0.0001
        self.device                  = "cuda:0"