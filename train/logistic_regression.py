import pickle, os
from data.dataset import generate_syntetic_data, generate_cobra_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from loguru import logger


def run_logistic_regression(cfg):

    if cfg.dataset == 'synthetic':
        logger.info("Generating synthetic data for Logistic regression")
        data, labels = generate_syntetic_data(cfg)
    elif cfg.dataset == "cobre":
        logger.info("Genereting Cobre data set for Logistic regression")
        data, labels, node_embeddings = generate_cobra_data(cfg)

    # lr = LogisticRegression(penalty='l1', solver='liblinear')
    lr = LogisticRegression(solver="liblinear", penalty="l2")

    scores = cross_val_score(
        lr,
        data.reshape(data.shape[0], -1),
        labels,
        cv=cfg.n_splits,
        scoring='f1_weighted',
        n_jobs=-1
        )

    logger.info(f"Logistic Regression scross val score : {scores.mean()}")

    ## if result folder does not exist, create it
    if not os.path.exists("./results/LR/"):
        os.makedirs("./results/LR/")

    with open("results/LR/log_reg_results.pkl", "wb") as f:
        pickle.dump(scores, f)

