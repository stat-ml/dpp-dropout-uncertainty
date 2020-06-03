import os
import pickle
import random
from pathlib import Path

import seaborn as sns
import pandas as pd
import torch
from torch.nn.functional import elu
import numpy as np
import matplotlib.pyplot as plt

from alpaca.uncertainty_estimator.masks import build_masks, DEFAULT_MASKS
from alpaca.analysis.metrics import uq_ll
from alpaca.model.ensemble import MLPEnsemble
from alpaca.uncertainty_estimator import build_estimator
from alpaca.analysis.metrics import get_uq_metrics

plt.rcParams['figure.facecolor'] = 'white'

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

folder = Path('./data/regression')
files = sorted([file for file in os.listdir(folder) if file.endswith('.pickle')])
files = [file for file in files if file.startswith('bos')]


def load_setup(file):
    print(file)
    with open(folder / 'log_exp.log', 'w') as f:
        f.write(f'{cnt} / {len(files)}')
    with open(folder / file, 'rb') as f:
        dct = pickle.load(f)
    config = dct['config']
    config['n_ue_runs'] = 1
    config['acc_percentile'] = .1
    state_dict = dct['state_dict']
    x_train, y_train, x_val, y_val, x_scaler, y_scaler = dct['data']

    ensemble = MLPEnsemble(
        config['layers'], n_models=config['n_ens'], activation = elu,
        reduction='mean')
    ensemble.load_state_dict(state_dict)

    model = ensemble.models[2]
    return model, ensemble, x_train, y_train, x_val, y_val, x_scaler, y_scaler


accumulate_ll = []
data = []


# np.random.shuffle(files)
for cnt, file in enumerate(files[:1]):
    model, ensemble, x_train, y_train, x_val, y_val, x_scaler, y_scaler = load_setup(file)

    x_val_tensor = torch.tensor(x_val)
    unscale = lambda y : y_scaler.inverse_transform(y)

    predictions = model(x_val_tensor.cuda()).cpu().detach().numpy()
    errors = predictions - y_val
    # rmse_single = np.sqrt(np.mean(np.square(errors)))
    # accumulate.append([file[:4], 'single', rmse_single])

    # predictions = ensemble(x_val_tensor.cuda()).cpu().detach().numpy()
    # errors = predictions - y_val
    # rmse_single = np.sqrt(np.mean(np.square(errors)))
    # accumulate.append([file[:4], 'ensemble', rmse_single])

    for mask_name in DEFAULT_MASKS:
        estimator = build_estimator(
            'mcdue_masked', model, nn_runs=100, keep_runs=True,
            dropout_mask=mask_name)
        ue =estimator.estimate(torch.Tensor(x_val).double().cuda())
        runs = estimator.last_mcd_runs()
        predictions = np.mean(estimator.last_mcd_runs(), axis=-1)
        errors = predictions - y_val[:, 0]
        rmse_mask = np.sqrt(np.mean(np.square(errors)))
        # accumulate.append([file[:4], mask_name, rmse_mask])
        ll = uq_ll(errors, ue)
        accumulate_ll.append([file[:4], mask_name, ll])


plt.figure(figsize=(10, 6))

accumulate2 = [record for record in accumulate_ll if record[1] !='decorrelating_sc']
df = pd.DataFrame(accumulate2, columns=['dataset', 'type', 'LL'])
sns.boxplot('dataset', 'LL', hue='type', data=df)
plt.savefig('ll_masks.png', dpi=150)

