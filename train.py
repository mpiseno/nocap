import os
import time
import math
import argparse

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.distributions import Normal

from nocap.config import default_config
from nocap.model import NoCAP
from nocap.dataset import AMASS_DS


def compute_nll(model, batch):
    # Output of the model are the parameters of a Gaussian. Mu and sigma are both
    # shape (B, T, 3), where B is the batch size and T is the sequence length.
    mu_, sigma_, _ = model(batch)

    # Model outputs conditional distributions over pose t given poses [1, t-1]
    # so we need to stagger the data.
    x = batch[:, 1:]
    mu, sigma = mu_[:, :-1], sigma_[:, :-1]

    # The objective is to minimize the negative log-likelihood of the data under
    # the model distribution: min -E_{x ~ p_data}[log p(x)].
    # Compute the log probability of the training data (the batch) under the 
    # model distribution.
    log_prob = (
        -1. * (x - mu).pow(2) / (2 * sigma.pow(2))
        - torch.log(math.sqrt(2 * math.pi) * sigma)
    ).sum(-1) # sum the last dimension. NOTE: Assumes independence. Is this ok?
    log_prob = log_prob.sum(-1) # sum time dimension
    loss = -1. * log_prob.mean()
    return loss


def train(model, optimizer, dataloader, epoch):
    model.train()
    total_loss = 0
    n_samples = 0
    for batch in dataloader:
        loss = compute_nll(model, batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_samples += len(batch)

    total_loss /= n_samples
    print(f'Epoch: {epoch} | Loss: {total_loss}')
    return total_loss


def run(args, config):
    model = NoCAP(config)

    train_data_path = os.path.join(args.data_dir, 'train_split.npy')
    train_dataset = AMASS_DS(train_data_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model, optimizer, train_dataloader, epoch)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=10)
    parser.add_argument('--data_dir', default='data/amass_processed')
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()

    config = default_config
    np.random.seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    run(args, config)