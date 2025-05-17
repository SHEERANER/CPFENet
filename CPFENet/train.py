import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import numpy as np
import os

from dataset import CTDataset  


def train_image(net, loss_fn, train_dataloader, valid_dataloader, gpu_ids, batch_size,
                num_epoch, lr, lr_min, optim='adam', init=False, scheduler_type='Cosine'):
    """
    Train a neural network model.

    Args:
        net: Neural network model to train.
        loss_fn: Loss function.
        train_dataloader: Training data loader.
        valid_dataloader: Validation data loader.
        gpu_ids: List of GPU IDs to use.
        batch_size: Batch size.
        num_epoch: Number of epochs.
        lr: Learning rate.
        lr_min: Minimum learning rate.
        optim: Optimizer type ('adam', 'sgd', 'adamw').
        init: Whether to initialize model weights.
        scheduler_type: Learning rate scheduler type.

    Returns:
        Training losses, training accuracies, validation losses, validation accuracies, and validation AUCs for each epoch.
    """

    # Weight initialization
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    # Set device (GPU/CPU)
    if gpu_ids:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        net = nn.DataParallel(net, device_ids=gpu_ids)
        net = net.to(device)
    else:
        device = torch.device('cpu')
        net = net.to(device)

    print(f'Training on: {device}')

    # Choose optimizer
    if optim == 'sgd':
        optimizer = SGD(net.parameters(), lr=lr)
    elif optim == 'adam':
        optimizer = Adam(net.parameters(), lr=lr)
    elif optim == 'adamw':
        optimizer = AdamW(net.parameters(), lr=lr)
    else:
        raise ValueError(f'Unsupported optimizer: {optim}')

    # Learning rate scheduler
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_type}')

    train_losses = []
    train_acces = []
    eval_losses = []
    eval_acces = []
    eval_auces = []
    best_auc = 0.0

    for epoch in range(num_epoch):
        print(f"—— Begin the {epoch + 1} epoch ——")

        # Training phase
        net.train()
        train_acc = 0.0
        for imgs, targets in tqdm(train_dataloader, desc='Training'):
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            num_correct = (preds == targets).sum().item()
            acc = num_correct / batch_size
            train_acc += acc

        scheduler.step()
        epoch_train_loss = loss.item()
        epoch_train_acc = train_acc / len(train_dataloader)
        print(f"Epoch: {epoch}, Loss: {epoch_train_loss}, Acc: {epoch_train_acc}")
        train_losses.append(epoch_train_loss)
        train_acces.append(epoch_train_acc)

        # Validation phase
        net.eval()
        eval_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = net(imgs)
                loss = loss_fn(outputs, targets)
                eval_loss += loss.item()

                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        eval_loss /= len(valid_dataloader)
        eval_acc = np.mean(np.array(all_targets) == np.array(all_preds))
        eval_auc = roc_auc_score(all_targets, all_preds)

        print(f"Validation: Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}, AUC: {eval_auc:.4f}")

        eval_losses.append(eval_loss)
        eval_acces.append(eval_acc)
        eval_auces.append(eval_auc)

        if eval_auc > best_auc:
            best_auc = eval_auc
            torch.save(net.state_dict(), 'best_auc.pth')

    return train_losses, train_acces, eval_losses, eval_acces, eval_auces