
from model.model import GCN
from model.logistic_regression import run_logistic_regression
from config import Config
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from model.train_utils import train_model
from data.dataset import generate_graph_dataset
from sklearn.model_selection import KFold
from loguru import logger
from argparse  import ArgumentParser


def run(cfg):

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
    logger.info(f"Accuracy train: {train_acc} val: {val_acc}")



if __name__  == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-th', '--adj_mat_threshold', type=float, default=0.5)
    args = parser.parse_args()
    logger.remove(0)

    for th in range(0, 10):
        th = th / 10
        Config.adj_mat_threshold = th

        logger.add(f"logdir/log_{Config.adj_mat_threshold}.log", level="INFO")
        logger.info("Experiment Configuration: \n{}".format(vars(Config)))

        ## TODO: add a test config atributes for attributes and values
        run(Config)
