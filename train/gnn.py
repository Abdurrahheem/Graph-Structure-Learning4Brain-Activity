from loguru import logger
from model.model import GCN
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from train.train_utils import train_model
from data.dataset import generate_graph_dataset
from sklearn.model_selection import KFold, train_test_split

def run_gnn(cfg):

    ## create data
    logger.info("Generating data...")
    dataset_list = generate_graph_dataset(cfg)

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
        weight_decay=cfg.weight_decay
    )

    ## split dataset to train and val splits
    train_set, val_set = train_test_split(
        dataset_list,
        test_size=cfg.val_size,
        random_state=cfg.seed
        )

    ## train model
    train_acc, val_acc, outs, labels = train_model(
        X_train=train_set,
        X_val=val_set,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        cfg=cfg,
    )