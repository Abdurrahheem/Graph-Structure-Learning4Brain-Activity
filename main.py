
from model.model import GCN
from model.logistic_regression import run_logistic_regression
from config import Config
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model.train_utils import train_model
from data.dataset import generate_graph_dataset
from sklearn.model_selection import KFold
from loguru import logger


def main(cfg):

    ## run logistinc regression for comparison
    logger.info("Running logistic regression...")
    run_logistic_regression(cfg)

    ## create data
    logger.info("Generating data...")
    dataset_list = generate_graph_dataset(cfg)

    ##create model
    model = GCN(
        hidden_channels = cfg.hidden_dim,
        N_rois = cfg.N_rois,
        output_size = cfg.classes,
        ).to(cfg.device)

    ## necessary perefirals
    optimizer = AdamW(model.parameters(), lr = cfg.lr,  weight_decay = cfg.weight_decay)
    criterion = CrossEntropyLoss()

    ## train model
    train_acc, val_acc, outs, labels = train_model(
        dataset_list[:8000],
        dataset_list[8000:],
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        cfg=cfg
    )
    print("Accuracy train: ", train_acc, " val:", val_acc)




if __name__  == '__main__':

    logger.info("Experiment Configuration: \n{}".format(vars(Config)))

    ## TODO: test config atributes for attributes and values
    main(Config)
