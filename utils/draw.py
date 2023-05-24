import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def draw_std(means, stds, taus, label, lr=None):

    plt.figure(figsize=(13,7))
    plt.plot(taus, means, '-o', label=label)
    if lr is not None:
        plt.plot(taus, lr, '-o', label="LR")
    plt.fill_between(taus, means - stds, means + stds, alpha=0.2)
    plt.xlabel(r"Adj Mat threshold $\tau$")
    plt.ylabel(label)
    plt.legend()
    plt.grid()
    plt.savefig(f"results/gnn/{label}.png")



def visualize_gnn():
    path = "results/gnn/results.pkl"
    path_lr = "results/LR/log_reg_results.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    with open(path_lr, "rb") as f:
        data_lr = pickle.load(f)

    print(data_lr)
    means, stds,  taus = [], [], []
    for k, v in data.items():
        # print(f"threshold: {k} | mean: {np.array(v).T.mean(0)} | std: {np.array(v).T.std(0)}")
        means.append(np.array(v).T.mean(0))
        stds.append(np.array(v).T.std(0))
        taus.append(k)

    means = np.array(means) * 100
    stds  = np.array(stds) * 100
    data_lr = np.array(data_lr) * 100

    print(means)
    draw_std(means[:, 1], stds[:, 1], np.arange(0.1, 0.8, 0.1), "Val_Accuracy")
    draw_std(means[:, 3], stds[:, 3], np.arange(0.1, 0.8, 0.1), "Val_F1_Score", lr=data_lr)

def visualize_gsl():
    path = "results/gsl/results.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    data = np.array(data)

    plt.figure(figsize=(13,7))
    plt.plot(data[:, 0], data[:, 1], "-o", label="Loss val_score")
    plt.xlabel("Epoches")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(f"results/gsl/F1_epoch.png")

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vis",
        choices=["gnn", "gsl"],
        required=True,
        help="Chose which results you want to visualize",
    )
    return parser.parse_args()

if __name__ == "__main__":

    args = get_args()
    visualize_gnn() if args.vis == "gnn" else visualize_gsl()