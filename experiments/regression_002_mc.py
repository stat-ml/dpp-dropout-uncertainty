from os import path
from argparse import ArgumentParser
import random
from pathlib import Path
import pickle

from alpaca.utils.datasets.builder import build_dataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from alpaca.ue.masks import BasicBernoulliMask, DPPMask
import alpaca.nn as ann
from alpaca.utils.model_builder import uncertainty_mode, inference_mode

from utils.metrics import uq_ll


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = Path('data/regression_7')

### params
lengthscale = 1e-2
T = 1000


def main(name, repeats, batch_size, sampler):
    print(name)
    manual_seed(42)
    dataset = build_dataset(name, val_split=0)
    x, y = dataset.dataset('train')
    N = x.shape[0] * 0.9  # train size

    #%%
    best_last, best_input, best_tau, best_dropout = select_params(x, y, N, batch_size, name, sampler)

    #%%
    vanilla_rmse = []
    vanilla_ll = []
    mc_rmse = []
    mc_ll = []

    for i in range(repeats):
        manual_seed(42 + i)
        print(i)
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)

        model = build_and_train(
            x_train,
            y_train,
            500,
            best_last,
            best_input,
            best_tau,
            best_dropout,
            N,
            batch_size,
            split_num=i,
            name=name,
            sampler=sampler
        )

        error, ll = rmse_ll(model, 1, x_test, y_test, y_scaler, tau=best_tau, dropout=False)
        vanilla_rmse.append(error)
        vanilla_ll.append(ll)

        error, ll = rmse_ll(model, T, x_test, y_test, y_scaler, tau=best_tau, dropout=True)
        mc_rmse.append(error)
        mc_ll.append(ll)

    print('vanilla')
    print(np.mean(vanilla_rmse), np.std(vanilla_rmse))
    print(np.mean(vanilla_ll), np.std(vanilla_ll))
    print(sampler)
    print(np.mean(mc_rmse), np.std(mc_rmse))
    print(np.mean(mc_ll), np.std(mc_ll))

    # plt.boxplot(vanilla_rmse)
    # plt.boxplot(mc_rmse)
    # plt.show()


class Network(nn.Module):
    def __init__(self, input_size, dropout_value, sampler, last_layer, input_layer):
        super().__init__()
        if sampler == 'mc':
            mask_class = BasicBernoulliMask
        else:
            mask_class = DPPMask

        base_size = 64
        self.last_layer = last_layer
        self.input_layer = input_layer

        self.dropout_1 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.fc1 = nn.Linear(input_size, base_size)

        self.dropout_2 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.fc2 = nn.Linear(base_size, 2*base_size)

        self.dropout_3 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.fc3 = nn.Linear(2*base_size, 2*base_size)

        self.dropout_4 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.fc4 = nn.Linear(2*base_size, 1)

    def forward(self, x):
        if self.input_layer:
            x = self.dropout_1(x)
        x = F.relu(self.fc1(x))

        if not self.last_layer:
            x = self.dropout_2(x)
        x = F.relu(self.fc2(x))

        if not self.last_layer:
            x = self.dropout_3(x)
        x = F.relu(self.fc3(x))

        x = self.dropout_4(x)
        x = self.fc4(x)
        return x


def loader(x_array, y_array, batch_size):
    dataset = TensorDataset(torch.Tensor(x_array), torch.Tensor(y_array))
    return DataLoader(dataset, batch_size=batch_size)


def rmse(values, predictions):
    return np.sqrt(np.mean(np.square(values - predictions)))


def rmse_ll(model, T, x_test, y_test, y_scaler, tau, dropout=True):
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    model.eval()
    if dropout:
        uncertainty_mode(model)
    else:
        inference_mode(model)

    with torch.no_grad():
        y_hat = np.array([
            y_scaler.inverse_transform(
                model(torch.Tensor(x_test).to(device).double()).cpu().numpy()
            )
            for _ in range(T)
        ])

    y_pred = np.mean(y_hat, axis=0)
    errors = np.abs(y_pred - y_test_unscaled)
    ue = np.std(y_hat, axis=0) + (1/tau)*y_scaler.scale_

    ll = uq_ll(errors, ue)
    return rmse(y_test_unscaled, y_pred), ll


def split_and_scale(x, y):
    # Load dat
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    # Scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, y_train, x_test, y_test, y_scaler


def build_and_train(
        x_train, y_train, epochs,
        last_layer, input_layer, tau, dropout_value,
        N, batch_size, split_num, name, sampler):
    if split_num is None:
        file_name = None
    else:
        file_name = save_dir / f"{name}_{sampler}_{split_num}.pt"

    reg = lengthscale**2 * (1 - dropout_value) / (2. * N * tau)
    train_loader = loader(x_train, y_train, batch_size)

    model = Network(x_train.shape[1], dropout_value, sampler, last_layer, input_layer).to(device).to(precision(sampler))
    if file_name and path.exists(file_name):
        model.load_state_dict(torch.load(file_name))
        model.eval()
    else:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=reg)
        criterion = nn.MSELoss()
        train_losses = []

        for epoch in range(epochs):
            losses = []
            for x_batch, y_batch in train_loader:
                preds = model(x_batch.to(device).to(precision(sampler)))
                optimizer.zero_grad()
                loss = criterion(y_batch.to(device).to(precision(sampler)), preds)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                train_losses.append(np.mean(losses))
        if file_name:
            torch.save(model.state_dict(), file_name)
    return model



def precision(sampler):
    if sampler == 'dpp':
        return torch.float64
    else:
        return torch.float32


def select_params(x, y, N, batch_size, name, sampler):
    file_name = save_dir/f"{name}_params_{sampler}.pickle"

    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        best_last = params['last_layer']
        best_input = params['input_layer']
        best_tau = params['tau']
        best_dropout = params['dropout']
    else:
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        results = []

        for local_last in [False]:
            for local_input in [True]:
                for local_tau in np.logspace(-4, 2, 14):
                    for local_dropout in [0.05, 0.2, 0.5]:
                        model = build_and_train(
                            x_train, y_train, 40,
                            local_last, local_input, local_tau, local_dropout,
                            N, batch_size, None, name, sampler
                        )
                        model.to(precision(sampler))
                        error, ll = rmse_ll(model, 200, x_test, y_test, y_scaler, dropout=True, tau=local_tau)
                        results.append((ll, error, (local_last, local_input, local_tau, local_dropout)))
                        print(results[-1])

        best_last, best_input, best_tau, best_dropout = sorted(results, key=lambda p: p[0])[-1][-1]
        print(best_last, best_input, best_tau, best_dropout)
        with open(file_name, 'wb') as f:
            params = {
                'tau': best_tau,
                'dropout': best_dropout,
                'last_layer': best_last,
                'input_layer': best_input
            }
            pickle.dump(params, f)
    return best_last, best_input, best_tau, best_dropout


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sampler', type=str, default='mc')
    args = parser.parse_args()

    main(args.name, args.repeats, args.batch_size, args.sampler)