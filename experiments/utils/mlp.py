from collections import OrderedDict
from itertools import count
import torch


class MLPEnsemble:
    def __init__(self, layers, n_models, reduction='mean', **kwargs):
        self.n_models = n_models
        self.models = [MLP(layers, **kwargs) for i in range(n_models)]
        self.reduction = reduction

    def state_dict(self):
        state_dict = OrderedDict({'{} model'.format(n): m.state_dict()
                                  for n, m in zip(count(), self.models)})
        return state_dict

    def load_state_dict(self, state_dict):
        for n, m in enumerate(self.models):
            m.load_state_dict(state_dict['{} model'.format(n)])

    def train(self):
        [m.train() for m in self.models]

    def eval(self):
        [m.eval() for m in self.models]

    def __call__(self, x, reduction='default', **kwargs):
        if 'dropout_mask' in kwargs and isinstance(kwargs['dropout_mask'], list):
            masks = kwargs.pop('dropout_mask')
            res = torch.stack([m(x, dropout_mask=dpm, **kwargs) for m, dpm in zip(self.models, masks)])
        else:
            res = torch.stack([m(x, **kwargs) for m in self.models])

        if reduction == 'default':
            reduction = self.reduction

        if reduction is None:
            res = res
        elif reduction == 'mean':
            res = res.mean(dim=0)
        elif reduction == 'nll':
            means = res[:, :, 0]
            sigmas = res[:, :, 1]
            res = torch.stack([means.mean(dim=0), sigmas.mean(dim=0) +
                               (means ** 2).mean(dim=0) - means.mean(dim=0) ** 2], dim=1)
        return res

    def _print_fit_status(self, n_model, n_models):
        print('Fit [{}/{}] model:'.format(n_model, n_models))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseMLP(nn.Module):
    def __init__(self, layer_sizes, activation, postprocessing=lambda x: x, device=None):
        super(BaseMLP, self).__init__()

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.layer_sizes = layer_sizes
        self.fcs = []
        for i, layer in enumerate(layer_sizes[:-1]):
            fc = nn.Linear(layer, layer_sizes[i+1])
            setattr(self, 'fc'+str(i), fc)  # to register params
            self.fcs.append(fc)
        self.postprocessing = postprocessing
        self.activation = activation
        self.double()
        self.to(self.device)

    def forward(self, x, dropout_rate=0, dropout_mask=None):
        x = self.activation(self.fcs[0](x))

        for layer_num, fc in enumerate(self.fcs[1:-1]):
            x = self.activation(fc(x))
            if dropout_mask is None:
                x = nn.Dropout(dropout_rate)(x)
            else:
                x = x*dropout_mask(x, dropout_rate, layer_num).double()
        x = self.fcs[-1](x)
        x = self.postprocessing(x)
        return x

    def evaluate(self, dataset, y_scaler=None):
        """ Return model losses for provided data loader """
        data_loader = loader(*dataset)
        with torch.no_grad():
            losses = []
            for points, labels in data_loader:
                points = points.reshape(-1, self.layer_sizes[0]).to(self.device)
                labels = labels.to(self.device)
                outputs = self(points)
                if y_scaler is not None:
                    outputs = torch.Tensor(y_scaler.inverse_transform(outputs.cpu()))
                    labels = torch.Tensor(y_scaler.inverse_transform(labels.cpu()))
                losses.append(self.criterion(outputs, labels).item())

        return sum(losses)/len(losses)

    def _print_status(self, epoch, epochs, loss, val_loss):
        print('Epoch [{}/{}], Loss: {:.4f}, Validation loss: {:.4f}'
              .format(epoch + 1, epochs, loss, val_loss))


class MLP(BaseMLP):
    def __init__(self, layer_sizes, postprocessing=None, loss=nn.MSELoss,
                 optimizer=None, activation=None, **kwargs):
        if postprocessing is None:
            postprocessing = lambda x: x

        if activation is None:
            activation = F.celu

        super(MLP, self).__init__(layer_sizes, activation=activation, postprocessing=postprocessing, **kwargs)

        self.criterion = loss()

        if optimizer is None:
            optimizer = {'type': 'Adadelta'}
        self.optimizer = self.init_optimizer(optimizer)

    def init_optimizer(self, optimizer):
        if isinstance(optimizer, dict):
            kwargs = optimizer.copy()
            opt_type = getattr(torch.optim, kwargs.pop('type'))
            kwargs['params'] = self.parameters()
            optimizer = opt_type(**kwargs)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                'optimizer must be either an Optimizer object or a dict, '
                'but got {}'.format(type(optimizer)))
        return optimizer