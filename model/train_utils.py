import torch
from torch_geometric.loader import DataLoader
from loguru import logger

def train(model, train_loader, criterion, optimizer, device):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out, x_pool = model(
                data.x.type(dtype=torch.float).to(device),
                data.edge_index.to(device),
                data.batch.to(device)
                )  # Perform a single forward pass.

        loss = criterion(out, data.y.to(device))  # Compute the loss.

        # Compute L1 loss component
     #     l1_weight = 1
     #     l1_parameters = []
     #     for parameter in model.parameters():
     #          l1_parameters.append(parameter.view(-1))
     #     l1 = l1_weight * model.compute_l1_loss(torch.cat(l1_parameters))

      # Add L1 loss component
     #     loss += l1

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, device):

    model.eval()

    correct = 0
    outs, labels =[], []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out,x_pool = model(data.x.type(dtype=torch.float).to(device),
                    data.edge_index.to(device),
                    data.batch.to(device))

        outs.append(x_pool.detach().cpu().numpy())
        labels.append(data.y.to(device).detach().cpu().numpy())

        pred = out.argmax(dim=1)  # Use the class with highest probability.

        correct += int((pred == data.y.to(device)).sum())  # Check against ground-truth labels.

    return correct / len(loader.dataset), outs, labels  # Derive ratio of correct predictions.



def train_model(X_train, X_val, model, optimizer, criterion, cfg):

    train_loader = DataLoader(X_train, batch_size=cfg.batch_size)
    val_loader   = DataLoader(X_val,   batch_size=cfg.batch_size)

    for epoch in range(1, cfg.epoch):
        logger.info(f'Epoch: {epoch}')

        train(model, train_loader, criterion, optimizer, device=cfg.device)

        train_acc, outs, labels  = test(model, train_loader, device=cfg.device)
        val_acc, outs, labels    = test(model, val_loader, device=cfg.device)

        logger.info(f'Train Accuracy: {train_acc:.4f}')
        logger.info(f'Val Accuracy: {val_acc:.4f}')

    return train_acc, val_acc, outs, labels