class DatasetConfig:
    def __init__ (self):
        ## data config
        self.dataset                 = "cobre"

        ## Other dataset parameters
        self.adj_mat_threshold       = 0.5

        ##
        self.val_size                = 0.2
        self.n_splits                = 7