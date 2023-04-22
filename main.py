from config import Config
from data import Data


def main(Config):
    ## create data
    dataset = Data(Config)
    print(len(dataset[0]))
    pass


if __name__  == '__main__':
    ## TODO: test config atributes for attributes and values
    main(Config)