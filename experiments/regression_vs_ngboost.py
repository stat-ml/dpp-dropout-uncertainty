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
from scipy.special import logsumexp

from alpaca.utils.ue_metrics import uq_ll
from alpaca.ue.masks import BasicBernoulliMask, DPPMask
import alpaca.nn as ann
from alpaca.utils.model_builder import uncertainty_mode, inference_mode


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


lengthscale = 1e-2
T = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = Path('data/regression_3')


def select_params(x, y, N, batch_size, name):
    file_name = save_dir/f"{name}_params.pickle"
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        best_tau = params['tau']
        best_dropout = params['dropout']
    else:
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        results = []
        for local_tau in np.logspace(-4, 3, 29):
            for local_dropout in [0.05]:
                model = build_and_train(
                    x_train, y_train, 50, local_tau, local_dropout, N, batch_size, None, name
                )
                error, ll = rmse_nll(model, 1, x_test, y_test, y_scaler, dropout=False, tau=local_tau)
                results.append((ll, error, (local_tau, local_dropout)))
                print(results[-1])

        best_tau, best_dropout = sorted(results, key=lambda p: p[0])[-1][-1]
        print(best_tau, best_dropout)
        with open(file_name, 'wb') as f:
            params = {'tau': best_tau, 'dropout': best_dropout}
            pickle.dump(params, f)
    return best_tau, best_dropout


def main(name, repeats, batch_size):
    manual_seed(42)
    dataset = build_dataset(name, val_split=0)
    x, y = dataset.dataset('train')
    N = x.shape[0] * 0.9  # train size

    #%%
    # best_tau, best_dropout = 0.01, 0.05
    best_tau, best_dropout = select_params(x, y, N, batch_size, name)

    #%%
    vanilla_rmse = []
    vanilla_ll = []
    mc_rmse = []
    mc_ll = []

    for i in range(repeats):
        print(i)
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        model = build_and_train(
            x_train, y_train, 400, best_tau, best_dropout, N, batch_size, split_num=i, name=name
        )

        error, ll = rmse_nll(model, 1, x_test, y_test, y_scaler, tau=best_tau, dropout=False)
        vanilla_rmse.append(error)
        vanilla_ll.append(ll)

        error, ll = rmse_nll(model, T, x_test, y_test, y_scaler, tau=best_tau, dropout=True)
        mc_rmse.append(error)
        mc_ll.append(ll)

    print('vanilla')
    print(np.mean(vanilla_rmse), np.std(vanilla_rmse))
    print(np.mean(vanilla_ll), np.std(vanilla_ll))
    print('mc')
    print(np.mean(mc_rmse), np.std(mc_rmse))
    print(np.mean(mc_ll), np.std(mc_ll))

    # plt.boxplot(vanilla_rmse)
    # plt.boxplot(mc_rmse)
    # plt.show()


class Network(nn.Module):
    def __init__(self, input_size, dropout_value):
        super().__init__()
        # mask_class = DPPMask
        # mask_class = BasicBernoulliMask
        # self.dropout_0 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.dropout_0 = nn.Dropout(dropout_value)
        self.fc1 = nn.Linear(input_size, 50)
        self.dropout_1 = nn.Dropout(dropout_value)
        # self.dropout_1 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.fc2 = nn.Linear(50, 50)
        # self.dropout_2 = ann.Dropout(dropout_rate=dropout_value, dropout_mask=mask_class())
        self.dropout_2 = nn.Dropout(dropout_value)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.dropout_0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_2(x)
        x = self.fc3(x)
        return x


def loader(x_array, y_array, batch_size):
    dataset = TensorDataset(torch.Tensor(x_array), torch.Tensor(y_array))
    return DataLoader(dataset, batch_size=batch_size)


def rmse(values, predictions):
    return np.sqrt(np.mean(np.square(values - predictions)))


def rmse_nll(model, T, x_test, y_test, y_scaler, tau, dropout=True):
    y_test_unscaled = y_scaler.inverse_transform(y_test)
    model.eval()
    if dropout:
        uncertainty_mode(model)
        # model.dropout_0.dropout_mask.reset()
        # model.dropout_1.dropout_mask.reset()
        # model.dropout_2.dropout_mask.reset()
        # model.train()
    else:
        inference_mode(model)
        # model.eval()

    with torch.no_grad():
        y_hat = np.array([
            y_scaler.inverse_transform(
                model(torch.Tensor(x_test).to(device).double()).cpu().numpy()
            )
            for _ in range(T)
        ])

    y_pred = np.mean(y_hat, axis=0)
    errors = np.abs(y_pred - y_test_unscaled)
    ue = np.std(y_hat, axis=0) + 1/tau
    ll = uq_ll(errors, ue)
    # ll = np.mean((logsumexp(-0.5 * tau * (y_test_unscaled[None] - y_hat)**2., 0) - np.log(T)
    #         - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)))
    return rmse(y_test_unscaled, y_pred), ll


def split_and_scale(x, y):
    # Load dat
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.01)

    # Scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    return x_train, y_train, x_test, y_test, y_scaler


def build_and_train(x_train, y_train, epochs, tau, dropout_value, N, batch_size, split_num, name):
    if split_num is None:
        file_name = None
    else:
        file_name = save_dir / f"{name}_{split_num}.pt"

    reg = lengthscale**2 * (1 - dropout_value) / (2. * N * tau)
    train_loader = loader(x_train, y_train, batch_size)

    model = Network(x_train.shape[1], dropout_value).to(device).double()
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
                preds = model(x_batch.to(device).double())
                optimizer.zero_grad()
                loss = criterion(y_batch.to(device).double(), preds)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                train_losses.append(np.mean(losses))
        if file_name:
            torch.save(model.state_dict(), file_name)
    return model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    main(args.name, args.repeats, args.batch_size)