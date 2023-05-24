import os
import torch
import copy
import pickle
from config.config import ConfigGSL
from data.dataset import generate_graph_dataset
from model.model_gsl import GCL, MLP_learner
from model.model import GCN
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils.utils_gsl import normalize, symmetrize
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from train.train_utils import test
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split
from loguru import logger



def loss_gcl(model, graph_learner, features, anchor_adj):

    # view 1: anchor graph
    # # if args.maskfeat_rate_anchor:
    # #     mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
    # #     features_v1 = features * (1 - mask_v1)
    # else:
    features_v1 = copy.deepcopy(features)

    z1, _ = model(features_v1, anchor_adj, 'anchor')

    # view 2: learned graph
    # if args.maskfeat_rate_learner:
    #     mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
    #     features_v2 = features * (1 - mask)
    # else:
    features_v2 = copy.deepcopy(features)

    learned_adj = graph_learner(features)
    # if not args.sparse:
    learned_adj = symmetrize(learned_adj)
    learned_adj = normalize(learned_adj, 'sym')

    z2, _ = model(features_v2, learned_adj, 'learner')

    # compute loss
    # if args.contrast_batch_size:
    #     node_idxs = list(range(features.shape[0]))
    #     # random.shuffle(node_idxs)
    #     batches = split_batch(node_idxs, args.contrast_batch_size)
    #     loss = 0
    #     for batch in batches:
    #         weight = len(batch) / features.shape[0]
    #         loss += model.calc_loss(z1[batch], z2[batch]) * weight
    # else:
    loss = model.calc_loss(z1, z2)

    return loss, learned_adj


def eval(graph_learner, dataset_list, cfg):

    train_index, val_index = train_test_split(
        dataset_list,
        test_size=cfg.val_size,
        shuffle=True,
        random_state=cfg.seed
    )

    train_loader = DataLoader(train_index, batch_size=cfg.batch_size)
    val_loader   = DataLoader(val_index, batch_size=cfg.batch_size)

    epoches = 100
    graph_learner.eval()
    graph_learner = graph_learner.to(cfg.device)

    ## define model
    model = GCN(cfg).to(cfg.device)
    model.train()

    ## necessary perefirals
    criterion = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr_gnn,
        weight_decay=cfg.weight_decay_eval if cfg.weight_decay_eval is not None else 0,
    )
    best_f1, best_acc = 0, 0
    for epoch in range(epoches):
        for data in train_loader:  # Iterate in batches over the training dataset.

            features = data.x.to(torch.float32)
            features = features.to(cfg.device)

            with torch.no_grad():
                learned_adj = graph_learner(features)
                edge_index, edge_weight = dense_to_sparse(learned_adj)
            assert learned_adj.shape[0] == features.shape[0]

            optimizer.zero_grad()  # Clear gradients.

            out, x_pool = model(
                data.x.type(dtype=torch.float).to(cfg.device),
                edge_index,
                data.batch.to(cfg.device),
                edge_weights=edge_weight
            )  # Perform a single forward pass.

            loss = criterion(out, data.y.to(cfg.device))  # Compute the loss.

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.

        acc, f1, _, _ = test_gsl(model, graph_learner, val_loader, cfg)

        if f1 > best_f1:
            best_f1 = max(best_f1, f1)
            best_acc = max(best_acc, acc)

    return best_acc, best_f1



def test_gsl(model, graph_learner, loader, cfg):

    graph_learner.eval()
    graph_learner = graph_learner.to(cfg.device)

    model.eval()
    model = model.to(cfg.device)

    outs, labels = [], []
    pd, gt = [], []
    for data in loader:  # Iterate in batches over the training/test dataset.

        features = data.x.to(torch.float32)
        features = features.to(cfg.device)

        with torch.no_grad():
            learned_adj = graph_learner(features)
            edge_index, edge_weight = dense_to_sparse(learned_adj)
            assert learned_adj.shape[0] == features.shape[0]

            out, x_pool = model(
                data.x.type(dtype=torch.float).to(cfg.device),
                edge_index,
                data.batch.to(cfg.device),
                edge_weights=edge_weight
            )  # Perform a single forward pass.

        ##TODO: why x_pool is used as an output?
        outs.append(x_pool.detach().cpu().numpy())
        labels.append(data.y.detach().cpu().numpy())

        pred = out.argmax(dim=1)  # Use the class with highest probability.

        pd.extend(pred.detach().cpu().numpy().tolist())
        gt.extend(data.y.detach().cpu().numpy().tolist())

    f1 = f1_score(pd, gt, average='weighted')
    accuracy = accuracy_score(pd, gt)
    return (
        accuracy,
        f1,
        outs,
        labels,
    )  # Derive ratio of correct predictions.



def train_gsl(cfg):

    dataset_list = generate_graph_dataset(cfg)

    train_loader = DataLoader(
        dataset_list,
        batch_size=cfg.batch_size,
        shuffle=True,
        )

    #TODO: move these to config
    graph_learner = MLP_learner(nlayers=cfg.nlayers,
                                isize=cfg.isize,
                                k=cfg.k,
                                knn_metric=cfg.knn_metric,
                                i=cfg.i,
                                sparse=False,
                                act=cfg.act_gl,
                                )
    model = GCL(nlayers=cfg.numlayers,
                in_dim=cfg.in_dim,
                hidden_dim= cfg.hidden_dim,
                emb_dim=cfg.emb_dim,
                proj_dim=cfg.proj_dim,
                dropout=cfg.dropout_gcl,
                dropout_adj=cfg.dropout_adj,
                sparse=None,
            )

    optimizer_cl = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr_proj,
        weight_decay=cfg.w_decay
    )
    optimizer_learner = torch.optim.Adam(
        graph_learner.parameters(),
        lr=cfg.lr_gl,
        weight_decay=cfg.w_decay
    )

    results = []
    best_acc, best_f1 = 0.0, 0.0
    for epoch in range(1, cfg.epoches):

        model.train()
        graph_learner.train()

        model = model.to(cfg.device)
        graph_learner = graph_learner.to(cfg.device)

        epoch_loss, couter = 0, 0
        for data in train_loader:  # Iterate in batches over the training dataset.

            features = data.x.to(cfg.device)
            features  = features.to(torch.float32)
            adjacency = data.edge_index.to(cfg.device)

            adjacency = to_dense_adj(adjacency).squeeze_()
            assert adjacency.shape[0] == features.shape[0]
            anchor_adj = copy.deepcopy(adjacency)

            loss, learned_adj = loss_gcl(model, graph_learner, features, anchor_adj)

            optimizer_cl.zero_grad()
            optimizer_learner.zero_grad()
            loss.backward()
            optimizer_cl.step()
            optimizer_learner.step()

            epoch_loss += loss.item()
            couter += 1

            # Structure Bootstrapping
            if (1 - cfg.tau) and (cfg.c == 0 or epoch % cfg.c == 0):
                anchor_adj = anchor_adj * cfg.tau + learned_adj.detach() * (1 - cfg.tau)


        if epoch % cfg.eval_freq == 0:

            acc, f1, = eval(
                            graph_learner=graph_learner,
                            dataset_list=dataset_list,
                            cfg=cfg
                        )
            results.append([epoch, epoch_loss / couter, f1, acc])

            if f1 > best_f1:
                best_f1 = max(best_f1, f1)
                best_acc = max(best_acc, acc)

        logger.info("Epoch {:05d} | CL Loss {:.4f} | F1: {:.4f} | Acc: {:.4f} ".format(epoch, epoch_loss / couter, best_f1, best_acc))


    ## if result folder does not exist, create it
    if not os.path.exists("./results/gsl/"):
        os.makedirs("./results/gsl/")

    ## save results
    with open("./results/gsl/results.pkl", "wb") as f:
        pickle.dump(results, f)