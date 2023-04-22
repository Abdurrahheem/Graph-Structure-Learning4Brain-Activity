import torch
import numpy as np
from torch_geometric.data import Data

def generate_syntetic_data(N_samples, N_rois, classes):

    ## currently only works for 2 classes
    assert classes == 2, "Currently only works for 2 classes :(("

    vals = np.random.standard_normal((N_samples, N_rois, N_rois))
    labels = np.random.randint(classes, size = N_samples)

    #TODO: make it working for more classes than 2
    for i in range(N_samples):
        if labels[i] == 0:
            vals[i,-1,-1] = 0.5 * np.random.random_sample()
        else:
            vals[i,-1,-1] = 0.5 * np.random.random_sample() + 0.5

    #TODO: make sure that symmetrified matrix values are corrcet
    ## symmetrify(upper and lower trinagular values are not the same anymore)
    for i in range(vals.shape[0]):
        vals[i, :,:] = (vals[i, :,:] + vals[i, :,:].T) / 2

    return vals, labels


def calculate_graph(vals, label, threshold):

    X = vals
    edge_indexes, edge_attres = [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > threshold:
                edge_indexes.append([i,j])
                edge_attres.append(X[i,j])

    return Data(x=torch.tensor(X),
                edge_index=torch.tensor(np.array(edge_indexes).T),
                y=torch.tensor([label]))


def get_data(config):

    vals, labels = generate_syntetic_data(config.N_samples, config.N_rois, config.classes)
    dataset      = calculate_graph(vals, labels, config.threshold)

    return dataset