import pickle
import numpy as np
import matplotlib.pyplot as plt


def draw(means, stds, taus, label, lr=None):

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
    print("Finished")

if __name__ == "__main__":

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
    draw(means[:, 1], stds[:, 1], np.arange(0.1, 0.8, 0.1), "Val_Accuracy")
    draw(means[:, 3], stds[:, 3], np.arange(0.1, 0.8, 0.1), "Val_F1_Score", lr=data_lr)
