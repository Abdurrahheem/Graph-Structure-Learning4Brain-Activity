import torch
from loguru import logger
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.

        out, x_pool = model(
            data.x.type(dtype=torch.float).to(device),
            data.edge_index.to(device),
            data.batch.to(device),
        )  # Perform a single forward pass.

        loss = criterion(out, data.y.to(device))  # Compute the loss.

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        # Compute L1 loss component
        l1_weight = 1
        l1_parameters = []
        for parameter in model.parameters():
            l1_parameters.append(parameter.view(-1))

        l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

        # Add L1 loss component
        loss += l1


def test(model, loader, device):

    model.eval()
    model = model.to(device)

    correct, all = 0, 0
    outs, labels = [], []
    pd, gt = [], []
    for data in loader:  # Iterate in batches over the training/test dataset.

        out, x_pool = model(
            data.x.type(dtype=torch.float).to(device),
            data.edge_index.to(device),
            data.batch.to(device),
        )

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


def train_model(cfg, X_train, X_val, model, optimizer, criterion, scheduler, verbose=False):

    train_loader = DataLoader(X_train, batch_size=cfg.batch_size)
    val_loader   = DataLoader(X_val, batch_size=cfg.batch_size)

    best_val_f1, best_val_acc, best_tr_f1, best_tr_acc  = 0, 0, 0, 0
    best_epoch = None
    for epoch in range(1, cfg.epoch):
        train(model, train_loader, criterion, optimizer, device=cfg.device)

        train_acc, tr_f1, outs, labels = test(model, train_loader, device=cfg.device)
        val_acc, vl_f1, outs, labels   = test(model, val_loader, device=cfg.device)

        if verbose:
            logger.info(
                f"Epoch {epoch} \ttrain acc: {train_acc:.4f}\tval acc: {val_acc:.4f}\n\t\t\t\t\t\t\t\t\t\t\ttrain f1:  {tr_f1:.4f}\tval f1:  {vl_f1:.4f}"
            )

        if vl_f1 > best_val_f1:

            best_tr_f1 = tr_f1
            best_val_f1 = vl_f1

            best_tr_acc = train_acc
            best_val_acc = val_acc

            best_epoch = epoch

    logger.info(
        f"Best Metrics -> Epoch {best_epoch} \ttrain acc: {best_tr_acc:.4f}\tval acc: {best_val_acc:.4f}\n\t\t\t\t\t\t\t\t\t\t\t\t\ttrain f1:  {best_tr_f1:.4f}\tval f1:  {best_val_f1:.4f}"
    )
    logger.info(
        f"Model parameters -> {model.params()}"
    )

    return best_tr_acc, best_val_acc, best_tr_f1, best_val_f1, outs, labels
