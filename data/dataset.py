import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from torch_geometric.data import Data

## write wrapper for measuring funtion execution time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} took: {end-start:0.3f} s")
        return result
    return wrapper

@measure_time
def generate_syntetic_data(N_samples, N_rois, classes):
    np.random.seed(12345)
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

@measure_time
def generate_cobra_data(cfg):

    ## check weather in dataset folder there is a folder with the name of the dataset
    data_path = Path.cwd() / "dataset" / "cobre" / "fmri" / "raw" / "aal"
    assert data_path.is_dir(), "THERE IS NO COBRE DATA SET"

    file_names = []
    for fn in data_path.iterdir():
        if "embed" not in str(fn):
            file_names.append(str(fn))

    assert (data_path.parent.parent.parent / "meta_data.tsv").is_file(), "THERE IS NO META DATA FILE"

    md = pd.read_csv(str(data_path.parent.parent.parent / "meta_data.tsv"), sep='\t')
    md = md[["Subjectid", "Dx"]]
    md = md.drop_duplicates()
    md = md.reset_index(drop=True)

    label_dict = {}
    for row in md.iterrows():
        label_dict[row[1]['Subjectid']] = row[1]['Dx']
    label_id = {j: i for i, j in enumerate(set([v for v in label_dict.values()]))}

    adjacencies, labels = [], []
    for fn in file_names:
        df = pd.read_csv(fn)
        sub_id = fn.split("/")[-1].split("-")[1].split(".")[0]
        A = df[df.columns[1:]].to_numpy()
        A = np.squeeze(A)
        adjacencies.append(A[None])
        labels.append(label_id[label_dict[sub_id]])
    adjacencies = np.concatenate(adjacencies, axis=0)
    return adjacencies, labels

def calculate_graph(vals, label, threshold):

    X = vals
    edge_indexes, edge_attres = [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > threshold:
                edge_indexes.append([i,j])

            ## TODO: what kind of edge attributes should be used?
            # edge_attres.append(X[i,j])

    return Data(x=torch.tensor(X),
                edge_index=torch.tensor(np.array(edge_indexes).T),
                y=torch.tensor([label]))


@measure_time
def generate_graph_dataset(cfg):

    if cfg.dataset.lower() == "cobre":
        logger.info("Generating Cobre Dataset....")
        vals, labels = generate_cobra_data(cfg)


    elif cfg.dataset.lower() == "syntetic":
        logger.info("Generating Syntetic Dataset....")
        vals, labels = generate_syntetic_data(cfg.N_samples, cfg.N_rois, cfg.classes)

    else:
        raise ValueError(f"Dataset: {cfg.dataset} not supported")

    logger.info("Thresholding Adjacency Matrices....")
    data_list = []
    for i in range(vals.shape[0]):
        data = calculate_graph(vals[i,:,:], labels[i],  cfg.adj_mat_threshold)
        data_list.append(data)

    return data_list