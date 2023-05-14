from config.cobre import CobreConfig
from config.synth import SynthConfig
from config.model import ModelConfig
from config.train import TrainConfig

class Config(CobreConfig, SynthConfig, ModelConfig, TrainConfig):

    ## data config
    # dataset                 = "cobre" #synthetic
    dataset                 = "synthetic"

    ## Other dataset parameters
    val_size                = 0.2
    adj_mat_threshold       = 0.2
    seed                    = 1234

    ## LogRegression for comparision
    LogRegression           = False


    def __init__(self):

        class_vars = vars(self.__class__)  # Get dictionary of class variables
        for var_name, var_value in class_vars.items():
            setattr(self, var_name, var_value)  # Create attribute with same name and value

        assert self.dataset in ["cobre", "synthetic"], f"{self.dataset} is not a valid dataset. \
                                                            Please choose from {['cobre', 'synthetic']}"
        if self.dataset == "cobre":
            print("initializing cobre")
            CobreConfig.__init__(self)

        elif self.dataset == "synthetic":
            print("initializing synthetic")
            SynthConfig.__init__(self)

        TrainConfig.__init__(self)
        ModelConfig.__init__(self)

