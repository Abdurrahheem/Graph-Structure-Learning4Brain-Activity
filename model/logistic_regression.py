from data.dataset import generate_syntetic_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from loguru import logger


def run_logistic_regression(cfg):
    data, labels = generate_syntetic_data(cfg.N_samples, cfg.N_rois,  cfg.classes)

    lr = LogisticRegression(penalty='l1', solver='liblinear')

    scores = cross_val_score(
        lr,
        data.reshape(cfg.N_samples, -1),
        labels,
        cv=5,
        scoring='accuracy')

    logger.info(f"Logistic Regression scross val score : {scores.mean()}")

