import torch
import copy
from config.config import Config
from gl_model import MLP_learner
from gst_model import GCL
from data.dataset import generate_graph_dataset
from utils.utils import set_seed, set_logger
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from gstl_utils import normalize, symmetrize, adj_to_index
from model.model import GCN
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from train.train_utils import test
from sklearn.metrics import f1_score, accuracy_score



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


def eval(graph_learner, data_loader, cfg):

    graph_learner.eval()
    graph_learner = graph_learner.to(cfg.device)

    ## define model
    model = GCN(cfg).to(cfg.device)
    model.train()

    ## necessary perefirals
    criterion = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0,
    )

    for data in data_loader:  # Iterate in batches over the training dataset.

        features = data.x.to(torch.float32)
        features = features.to(cfg.device)
        adjacency = data.edge_index
        # adjacency = to_dense_adj(data.edge_index).squeeze_()
        # assert adjacency.shape[0] == features.shape[0]

        with torch.no_grad():
            learned_adj = graph_learner(features)
            edge_index, edge_weight = dense_to_sparse(learned_adj)
            # edge_index, edge_weight = adj_to_index(learned_adj)
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

    acc, f1, out, labels = test_gsl(model, graph_learner, data_loader,  cfg.device)
    print(f"Accuracy: {acc}, f1: {f1}")


def test_gsl(model, graph_learner, loader, device):

    graph_learner.eval()
    model.eval()
    model = model.to(device)
    graph_learner = graph_learner.to(device)

    correct, all = 0, 0
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
        labels.append(data.y.to(device).detach().cpu().numpy())

        # print(out.shape)
        pred = out.argmax(dim=1)  # Use the class with highest probability.

        pd.extend(pred.detach().cpu().numpy().tolist())
        gt.extend(data.y.to(device).detach().cpu().numpy().tolist())

    f1 = f1_score(pd, gt, average='weighted')
    accuracy = accuracy_score(pd, gt)
    return (
        accuracy,
        f1,
        outs,
        labels,
    )  # Derive ratio of correct predictions.



if __name__ == "__main__":
    cfg = Config()

    set_seed(cfg)
    # set_logger(cfg)


    epoches         = 10
    c               = 0
    tau             = 0.9999
    batch_size      = 20
    lr              = 0.01
    w_decay         = 0.0
    eval_freq       = 1
    device          = cfg.device

    dataset_list = generate_graph_dataset(cfg)

    print(len(dataset_list))
    train_loader = DataLoader(dataset_list[:130], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_list[130:], batch_size=batch_size, shuffle=True)

    graph_learner = MLP_learner(nlayers=2,
                                isize=150,
                                k=30,
                                knn_metric="cosine_sim",
                                i=6,
                                sparse=False,
                                act='relu',
                                )
    model = GCL(nlayers=2,
                in_dim=150,
                hidden_dim= 150 // 2,
                emb_dim=50,
                proj_dim=30,
                dropout=0.5,
                dropout_adj=0.2,
                sparse=None,
            )

    optimizer_cl = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=w_decay
    )
    optimizer_learner = torch.optim.Adam(
        graph_learner.parameters(),
        lr=lr,
        weight_decay=w_decay
    )

    for epoch in range(1, epoches):

        model.train()
        graph_learner.train()

        model = model.to(device)
        graph_learner = graph_learner.to(device)

        epoch_loss, couter = 0, 0
        for data in train_loader:  # Iterate in batches over the training dataset.

            features = data.x.to(device)
            features  = features.to(torch.float32)
            adjacency = data.edge_index.to(device)

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
            if (1 - tau) and (c == 0 or epoch % c == 0):
                anchor_adj = anchor_adj * tau + learned_adj.detach() * (1 - tau)

        print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, epoch_loss / couter))

        if epoch % eval_freq == 0:

            eval(
                graph_learner=graph_learner,
                data_loader=train_loader,
                cfg=cfg
            )
