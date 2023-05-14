from loguru import logger
from config.config import Config
from train.gnn import run_gnn
from train.logistic_regression import run_logistic_regression
from tabulate import tabulate
import pprint

def get_config_dict(cfg):

    config_dict = {}

    all_attributes = dir(cfg.__class__)
    for attribute in all_attributes:
        if attribute.startswith("__"):
            continue
        value = getattr(cfg, attribute)
        config_dict[attribute] = value

    return  config_dict

def fancy_dict(cfg):
    table_header = ["keys", "values"]
    exp_table = [
        (str(k), pprint.pformat(v))
        # for k, v in get_config_dict(cfg).items()
        for k, v in vars(cfg).items()
        if not k.startswith("_")
    ]
    return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

def main(cfg):
    ##  logistinc regression for comparison
    if cfg.LogRegression:
        logger.info("Training Logistic Regression...")
        run_logistic_regression(cfg)

    logger.info("Training GNN...")
    run_gnn(cfg)

if __name__ == "__main__":
    cfg = Config()

    logger.info(f"\n{fancy_dict(cfg)}")

    main(cfg)
