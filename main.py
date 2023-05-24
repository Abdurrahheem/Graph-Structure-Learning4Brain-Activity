import argparse
from loguru import logger
from train.gnn import train_gnn
from train.gsl import train_gsl
from train.lr import run_logistic_regression
from utils.utils import set_seed, set_logger
from config.config import Config, ConfigGSL

def main(cfg, args):

    ## logistic regression for comparison
    if args.LogReg:
        logger.info("Training Logistic Regression...")
        run_logistic_regression(cfg)

    if args.model == "gnn":
        logger.info("Training GNN...")
        train_gnn(cfg)
    else:
        logger.info("Training GNN...")
        train_gsl(cfg)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        choices=["gnn", "gsl"],
        required=True,
        help="Choose model type you want to train. GNN or Graph Structure Learner",
    )

    parser.add_argument(
        "-LR",
        "--LogReg",
        type=bool,
        default=True,
        help="Train Logistice Regression Model",
    )

    parser.add_argument(
        "-d",
        "--device",
        default='cuda',
        choices=["cpu", "cuda"],
        help="Device to train on",
    )

    return parser.parse_args()


def merge(cfg, args):
    cfg.device = args.device
    return cfg

if __name__ == "__main__":

    args = get_args()
    cfg = Config() if args.model == "gnn" else ConfigGSL()
    cfg = merge(cfg, args)

    set_seed(cfg)
    set_logger(cfg)

    main(cfg, args)
