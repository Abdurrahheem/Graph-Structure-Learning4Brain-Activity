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
def generate_syntetic_data(cfg):
    np.random.seed(cfg.seed)
    ## currently only works for 2 classes
    assert cfg.classes == 2, "Currently only works for 2 classes :(("

    vals = np.random.standard_normal((cfg.N_samples, cfg.N_rois, cfg.N_rois))
    labels = np.random.randint(cfg.classes, size = cfg.N_samples)

    #TODO: make it working for more classes than 2
    for i in range(cfg.N_samples):
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
    file_names_embed = []
    for fn in data_path.iterdir():
        if "embed" not in str(fn):
            file_names.append(str(fn))
        else:
            file_names_embed.append(str(fn))

    assert (data_path.parent.parent.parent / "meta_data.tsv").is_file(), "THERE IS NO META DATA FILE"

    md = pd.read_csv(str(data_path.parent.parent.parent / "meta_data.tsv"), sep='\t')
    md = md[["Subjectid", "Dx"]]
    md = md.drop_duplicates()
    md = md.reset_index(drop=True)

    label_dict = {}
    for row in md.iterrows():
        label_dict[row[1]['Subjectid']] = row[1]['Dx']

    ## filter out "Schizoaffective" class are only small amout of values there
    label_dict = {k: v for k, v in label_dict.items() if v != "Schizoaffective"}
    label_id = {j: i for i, j in enumerate(sorted(set([v for v in label_dict.values()])))}

    adjacencies, node_embeddings, labels = [], [], []
    for fn, fn_emb in zip(file_names, file_names_embed):
        df = pd.read_csv(fn)
        df_emb = pd.read_csv(fn_emb, index_col=False)

        sub_id = fn.split("/")[-1].split("-")[1].split(".")[0]

        if sub_id not in label_dict:
            continue

        df_emb = df_emb.loc[:, ~df_emb.columns.str.contains('^Unnamed')]
        A_emb = df_emb.to_numpy().T

        A = df[df.columns[1:]].to_numpy()
        A = np.squeeze(A)

        node_embeddings.append(A_emb[None])
        adjacencies.append(A[None])
        labels.append(label_id[label_dict[sub_id]])

    adjacencies = np.concatenate(adjacencies, axis=0)
    node_embeddings = np.concatenate(node_embeddings, axis=0)
    return adjacencies, labels, node_embeddings

def calculate_graph(vals, label, threshold, node_embeddings=None):

    if node_embeddings is not None:
        assert vals.shape[0] == node_embeddings.shape[0]
        assert node_embeddings.shape[1] > 0

    X = vals
    edge_indexes, edge_attres, = [], []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > threshold:
                edge_indexes.append([i,j])

            ## TODO: what kind of edge attributes should be used?
            # edge_attres.append(X[i,j])
    # print(node_embeddings.shape, np.array(edge_indexes).T.shape)

    if node_embeddings is not None:
        return Data(
            x=torch.tensor(node_embeddings),
            edge_index=torch.tensor(np.array(edge_indexes).T),
            y=torch.tensor([label])
            )
    else:
        return Data(
            x=torch.tensor(X),
            edge_index=torch.tensor(np.array(edge_indexes).T),
            y=torch.tensor([label])
            )


@measure_time
def generate_graph_dataset(cfg):

    if cfg.dataset.lower() == "cobre":
        logger.info("Generating Cobre Dataset....")
        vals, labels, node_embeddings = generate_cobra_data(cfg)


    elif cfg.dataset.lower() == "synthetic":
        logger.info("Generating Syntetic Dataset....")
        vals, labels = generate_syntetic_data(cfg)

    else:
        raise ValueError(f"Dataset: {cfg.dataset} not supported")

    logger.info("Thresholding Adjacency Matrices....")
    data_list = []
    for i in range(vals.shape[0]):
        if cfg.dataset.lower() == "cobre" and cfg.use_node_embeddings:
            data = calculate_graph(vals[i,:,:], labels[i],  cfg.adj_mat_threshold, node_embeddings[i, :, :])
        else:
            data = calculate_graph(vals[i,:,:], labels[i],  cfg.adj_mat_threshold)
        data_list.append(data)

    return data_list