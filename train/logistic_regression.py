from data.dataset import generate_syntetic_data, generate_cobra_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from loguru import logger


def run_logistic_regression(cfg):

    if cfg.dataset == 'synthetic':
        logger.info("Generating synthetic data for Logistic regression")
        data, labels = generate_syntetic_data(cfg.N_samples, cfg.N_rois,  cfg.classes)
    elif cfg.dataset == "cobre":
        logger.info("Genereting Cobre data set for Logistic regression")
        data, labels = generate_cobra_data(cfg)

    lr = LogisticRegression(penalty='l1', solver='liblinear')

    scores = cross_val_score(
        lr,
        data.reshape(data.shape[0], -1),
        labels,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1
        )

    logger.info(f"Logistic Regression scross val score : {scores.mean()}")
    exit(0);

