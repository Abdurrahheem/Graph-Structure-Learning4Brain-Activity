from loguru import logger
from config.config import Config
from train.gnn import run_gnn
from train.logistic_regression import run_logistic_regression
from utils.utils import set_seed, set_logger

def main(cfg):
    ##  logistinc regression for comparison
    if cfg.LogRegression:
        logger.info("Training Logistic Regression...")
        run_logistic_regression(cfg)

    logger.info("Training GNN...")
    run_gnn(cfg)

if __name__ == "__main__":
    cfg = Config()

    set_seed(cfg)
    set_logger(cfg)

    main(cfg)
