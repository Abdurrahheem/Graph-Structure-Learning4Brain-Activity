import torch
import datetime
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn

from tabulate import tabulate
from loguru import logger

def set_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    cudnn.deterministic = True


def set_logger(cfg):
    logger.add(f"logdir/{datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')}.log", backtrace=True, diagnose=True)
    logger.info(f"\n{fancy_dict(cfg)}")

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