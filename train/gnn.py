import numpy as np
import pickle
import os

from loguru import logger
from model.model import GCN
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from train.train_utils import train_model
from data.dataset import generate_graph_dataset
from sklearn.model_selection import KFold, train_test_split

def run_gnn(cfg):

    ## create data
    data = {}
    for threshold in np.arange(0.1, 0.8, 0.1):

        cfg.adj_mat_threshold = threshold

        logger.info("Generating data...")
        logger.info(f"Adjacency matrix threshold: {threshold}")

        dataset_list = generate_graph_dataset(cfg)

        mean_tr_acc, mean_vl_acc, mean_tr_f1, mean_vl_f1 = [], [], [], []
        kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_list)):

            logger.info(f"Fold {fold + 1}")
            train_set = [dataset_list[i] for i in train_idx]
            val_set   = [dataset_list[i] for i in val_idx]

            ##create model
            model = GCN(
                hidden_channels=cfg.hidden_dim,
                N_rois=cfg.N_rois,
                output_size=cfg.classes,
            ).to(cfg.device)

            ## necessary perefirals
            criterion = CrossEntropyLoss()
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0,
            )
            scheduler = None

            ## train model
            train_acc, val_acc, tr_f1, vl_f1, outs, labels = train_model(
                cfg=cfg,
                X_train=train_set,
                X_val=val_set,
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                verbose=cfg.verbose,
            )

            mean_tr_acc.append(train_acc)
            mean_vl_acc.append(val_acc)
            mean_tr_f1.append(tr_f1)
            mean_vl_f1.append(vl_f1)

        logger.info("====================")
        logger.info(f"Mean Train Accuracy:\t{sum(mean_tr_acc)/len(mean_tr_acc)}")
        logger.info(f"Mean Val Accuracy:\t\t{sum(mean_vl_acc)/len(mean_vl_acc)}")
        logger.info(f"Mean Train F1:\t\t{sum(mean_tr_f1)/len(mean_tr_f1)}")
        logger.info(f"Mean Val F1:\t\t{sum(mean_vl_f1)/len(mean_vl_f1)}")

        data[threshold] = [mean_tr_acc, mean_vl_acc, mean_tr_f1, mean_vl_f1]

    ## if result folder does not exist, create it
    if not os.path.exists("./results/gnn/"):
        os.makedirs("./results/gnn/")

    ## save results
    with open("./results/gnn/results.pkl", "wb") as f:
        pickle.dump(data, f)