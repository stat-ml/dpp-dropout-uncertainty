import os
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from torch import nn
import torch

from alpaca.utils.datasets.builder import build_dataset
from ngboost import NGBRegressor

from regression_002_mc import manual_seed, split_and_scale, uq_ll, Network, loader, device, rmse


ensemble_size = 10
save_dir = Path('data/regression_5/ensembles')
dropout_value = 0.02
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

reg = 1e-4


def build_and_train(
        x_train, y_train, epochs, dropout_value,
        batch_size, split_num, model_num, name
    ):

    file_name = save_dir / f"{name}_{split_num}_{model_num}.pt"
    train_loader = loader(x_train, y_train, batch_size)

    model = Network(x_train.shape[1], dropout_value, 'mc', False, False).to(device)
    if file_name and os.path.exists(file_name):
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
                preds = model(x_batch.to(device))
                optimizer.zero_grad()
                loss = criterion(y_batch.to(device), preds)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 10 == 0:
                train_losses.append(np.mean(losses))
        if file_name:
            torch.save(model.state_dict(), file_name)
    return model


def rmse_ll(x_test, y_test, y_scaler, models):
    y_test_unscaled = y_scaler.inverse_transform(y_test)

    for model in models:
        model.eval()

    with torch.no_grad():
        y_hat = np.array([
            y_scaler.inverse_transform(
                model(torch.Tensor(x_test).to(device)).cpu().numpy()
            )
            for model in models
        ])

    y_pred = np.mean(y_hat, axis=0)
    errors = np.abs(y_pred - y_test_unscaled)
    ue = np.std(y_hat, axis=0)
    ll = uq_ll(errors, ue)
    return rmse(y_test_unscaled, y_pred), ll


def main(name, repeats):
    print(name)
    manual_seed(42)
    dataset = build_dataset(name, val_split=0)
    x, y = dataset.dataset('train')
    result_errors = []
    result_ll = []
    for i in range(repeats):
        manual_seed(42 + i)
        models = []
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        for model_num in range(ensemble_size):
            x_temp, _, y_temp, _ = train_test_split(x_train, y_train, train_size=0.8)  # kind of bootstrap
            model = build_and_train(
                x_temp, y_temp, epochs=400, dropout_value=dropout_value, batch_size=128,
                split_num=i, model_num=model_num, name=name
            )
            models.append(model)

        rmse, ll = rmse_ll(x_test, y_test, y_scaler, models)
        result_errors.append(rmse)
        result_ll.append(ll)

    print(np.mean(result_errors), np.std(result_errors))
    print(np.mean(result_ll), np.std(result_ll))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()

    main(args.name, args.repeats)