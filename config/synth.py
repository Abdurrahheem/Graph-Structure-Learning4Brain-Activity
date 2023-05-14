class SynthConfig:

    N_samples               = 10000
    N_rois                  = 100
    classes                 = 2

    def __init__ (self):
        class_vars = vars(self.__class__)  # Get dictionary of class variables
        for var_name, var_value in class_vars.items():
            setattr(self, var_name, var_value)  # Create attribute with same name and value