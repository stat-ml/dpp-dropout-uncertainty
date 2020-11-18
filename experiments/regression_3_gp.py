#%%
import math
from os import path
from argparse import ArgumentParser
import random
from pathlib import Path
import pickle

import gpytorch
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as mse

from alpaca.utils.datasets.builder import build_dataset
from alpaca.utils.ue_metrics import uq_ll

from regression_002_mc import manual_seed, split_and_scale

device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = Path('data/regression_5')


def rmse_ll(true_y, prediction, uncertainty, y_scaler):
    errors = np.abs(true_y - prediction)
    ll = uq_ll(errors, uncertainty) * y_scaler.scale_[0]
    rms_error = np.square(
        mse(
            y_scaler.inverse_transform(true_y),
            y_scaler.inverse_transform(prediction)
        )
    )
    return rms_error, ll


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=x_train.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main(name, repeats):
    #%%
    manual_seed(42)
    dataset = build_dataset(name, val_split=0)
    x, y = dataset.dataset('train')
    N = x.shape[0] * 0.9  # train size
    # for i in range(repeats):
    result_errors = []
    result_ll = []

    for run in range(repeats):
        print(run, end=' ', flush=True)
        manual_seed(42 + run)
        x_train, y_train, x_test, y_test, y_scaler = split_and_scale(x, y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

        # train gp
        x_train_ = torch.Tensor(x_train).cuda()
        x_test_ = torch.Tensor(x_test).cuda()
        y_train_ = torch.Tensor(y_train[:, 0]).cuda()
        y_test_ = torch.Tensor(y_test[:, 0]).cuda()

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(x_train_, y_train_, likelihood)

        # Find optimal model hyperparameters
        model.train().cuda()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iter = 1_000
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train_)
            # # Calc loss and backprop gradients

            loss = -mll(output, y_train_)
            loss.backward()
            # if i % 20 == 0:
                # print('Iter %d/%d - Loss: %.3f   lengthscale: %s   noise: %.3f' % (
                #     i + 1, training_iter, loss.item(),
                #     model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()[0, :3],
                #     model.likelihood.noise.item()
                # ))
            optimizer.step()

        # calculate results
        x_test_.cuda()
        model.eval().cuda()
        likelihood.eval()
        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            model(x_test_)
            observed_pred = likelihood(model(x_test_))
            mean = observed_pred.mean
            uq = np.sqrt((observed_pred._covar.diag()).cpu().numpy())
            error, ll = rmse_ll(y_test[:, 0], mean.cpu().numpy(), uq, y_scaler)
            result_errors.append(error)
            result_ll.append(ll)

    print()
    print(name)
    print(np.mean(result_errors), np.std(result_errors))
    print(np.mean(result_ll), np.std(result_ll))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()

    main(args.name, args.repeats)