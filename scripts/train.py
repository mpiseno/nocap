import os
import sys
import shutil
import time
import math
import argparse
import pathlib

import tqdm
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

from nocap.config import default_config
from nocap.model import NoCAP
from nocap.dataset import NoCAP_DS


LOG_DIR = 'logs'


def compute_nll(model, batch):
    # Output of the model are the parameters of a Gaussian. Mu and sigma are both
    # shape (B, T, 3), where B is the batch size and T is the sequence length.
    mu_, sigma_, _ = model(batch)

    # Model outputs conditional distributions over pose t given poses [1, t-1]
    # so we need to stagger the data.
    x = batch[:, 1:, :3] # slice out joint encoding
    mu, sigma = mu_[:, :-1], sigma_[:, :-1]

    # The objective is to minimize the negative log-likelihood of the data under
    # the model distribution: min -E_{x ~ p_data}[log p(x)].
    # Compute the log probability of the training data (the batch) under the 
    # model distribution.
    log_prob = (
        -1. * (x - mu).pow(2) / (2 * sigma.pow(2))
        - torch.log(math.sqrt(2 * math.pi) * sigma)
    ).sum(-1)
    log_prob = log_prob.sum(-1) # sum time dimension
    loss = -1. * log_prob.mean()
    return loss


def train(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in tqdm.tqdm(dataloader):
        batch = batch.to(device)
        loss = compute_nll(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += len(batch)

    total_loss /= n_samples
    print(f'Epoch: {epoch} | Train Loss: {total_loss}')
    return total_loss


def test(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0
    n_samples = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            batch = batch.to(device)
            loss = compute_nll(model, batch)

            total_loss += loss.item()
            n_samples += len(batch)

    total_loss /= n_samples
    print(f'Epoch: {epoch} | Val Loss: {total_loss}')
    return total_loss


def save(save_dir, model, config, epoch):
    save_path = os.path.join(save_dir, f'ckpt_epoch={epoch}.pt')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': config
    }
    torch.save(save_dict, save_path)


def log(log_dir, writer, stats, epoch):
    for key, val in stats.items():
        writer.add_scalar(key, val, epoch)


def run(args, config):
    log_dir = os.path.join(LOG_DIR, args.exp_name)
    save_dir = os.path.join(log_dir, 'checkpoints')
    writer = None
    log_and_save = not args.disable_logging
    if log_and_save:
        if os.path.exists(log_dir):
            response = input(f'Log directory {log_dir} exists. Overwrite? (y/n): ')
            if response.lower() != 'y':
                print(f'Exiting...')
                sys.exit(0)
            else:
                shutil.rmtree(log_dir)

        pathlib.Path(save_dir).mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = NoCAP(config).to(device)

    #train_data_path = os.path.join(args.data_dir, 'train_split.npz')
    train_data_path = 'data/amass_processed/by_file/0005_Jogging001_stageii.npz'
    train_dataset = NoCAP_DS(train_data_path)

    #val_data_path = os.path.join(args.data_dir, 'val_split.npz')
    #val_dataset = NoCAP_DS(val_data_path)
    val_dataset = train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss, best_val_epoch = 1e10, None
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model, optimizer, train_dataloader, device, epoch)
        stats = {
            'train_nll': train_loss
        }

        if epoch % args.val_freq == 0:
            val_loss = test(model, val_dataloader, device, epoch)
            stats['val_nll'] = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_epoch = epoch

        if log_and_save and epoch % args.save_freq == 0:
            save(save_dir, model, config, epoch)
        
        if log_and_save:
            log(log_dir, writer, stats, epoch)

    if log_and_save:
        save(save_dir, model, config, epoch='final')

    print(f'Best val loss: {best_val_loss}, epoch: {best_val_epoch}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_ckpts', type=bool, default=True)
    parser.add_argument('--disable_logging', action='store_true', default=False)
    parser.add_argument('--exp_name', type=str, default='default')
    args = parser.parse_args()

    config = default_config
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    run(args, config)