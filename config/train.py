class TrainConfig:
    batch_size              = 30
    epoch                   = 100
    lr                      = 0.0001
    weight_decay            = 0.0001
    device                  = "cuda:0"

    def __init__ (self):
        class_vars = vars(self.__class__)  # Get dictionary of class variables
        for var_name, var_value in class_vars.items():
            setattr(self, var_name, var_value)  # Create attribute with same name and value