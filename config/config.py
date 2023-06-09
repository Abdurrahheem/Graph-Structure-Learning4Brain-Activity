from config.cobre import CobreConfig
from config.synth import SynthConfig
from config.model import ModelConfig, MLPLearnerConfig, GCLConfig
from config.train import TrainConfig, TrainConfigGSL
from config.dataset import DatasetConfig

class Config(CobreConfig, SynthConfig, ModelConfig, TrainConfig, DatasetConfig):
    ## Other dataset parameters
    seed                    = 1234

    ## LogRegression for comparision
    LogRegression           = True

    def __init__(self):

        class_vars = vars(self.__class__)  # Get dictionary of class variables
        for var_name, var_value in class_vars.items():
            setattr(self, var_name, var_value)  # Create attribute with same name and value

        TrainConfig.__init__(self)
        ModelConfig.__init__(self)
        DatasetConfig.__init__(self)


        if self.dataset == "cobre":
            print("initializing cobre")
            CobreConfig.__init__(self)

        elif self.dataset == "synthetic":
            print("initializing synthetic")
            SynthConfig.__init__(self)


        assert self.dataset in ["cobre", "synthetic"], f"{self.dataset} is not a valid dataset. \
                                                            Please choose from {['cobre', 'synthetic']}"

class ConfigGSL(
    CobreConfig,
    DatasetConfig,
    TrainConfigGSL,
    ModelConfig,
    MLPLearnerConfig,
    GCLConfig,
    ):
    seed = 1234

    def __init__(self):

        class_vars = vars(self.__class__)  # Get dictionary of class variables
        for var_name, var_value in class_vars.items():
            setattr(self, var_name, var_value)  # Create attribute with same name and value

        TrainConfigGSL.__init__(self)
        DatasetConfig.__init__(self)
        ModelConfig.__init__(self)
        MLPLearnerConfig.__init__(self)
        GCLConfig.__init__(self)

        if self.dataset == "cobre":
            assert self.dataset in ["cobre", "synthetic"], f"{self.dataset} is not a valid dataset. \
                                                            Please choose from {['cobre', 'synthetic']}"
            CobreConfig.__init__(self)