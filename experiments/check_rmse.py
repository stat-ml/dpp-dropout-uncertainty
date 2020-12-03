import torch
import pickle
import os
from pathlib import Path
import numpy as np
from torch.nn.functional import elu
from utils.mlp import MLPEnsemble





datasets = [
    'boston_housing', 'concrete', 'energy_efficiency',
    'kin8nm', 'naval_propulsion', 'ccpp', 'red_wine',
    'yacht_hydrodynamics'
]

folder = Path('data/regression')


def main():
    for dataset in datasets:
        get_rmse(dataset)


def get_rmse(dataset):
    filenames = [name for name in os.listdir(folder) if name.startswith(dataset[:4])]
    print(dataset)
    results = []

    for filename in filenames:
        with open(folder / filename, 'rb') as f:
            dct = pickle.load(f)
            x_train, y_train, x_val, y_val, x_scaler, y_scaler = dct['data']
            net = build_model(dct['state_dict'], x_train.shape[1])

            x_tensor = torch.DoubleTensor(x_val).cuda()
            preds = net(x_tensor).detach().cpu().numpy()
            rmse = np.sqrt(np.mean(np.square(
                y_scaler.inverse_transform(preds) -
                y_scaler.inverse_transform(y_val)
            )))
            results.append(rmse)
    print(np.mean(results))


def build_model(state_dict, inp_size):
    config = {
        'layers': [inp_size, 128, 128, 64, 1],
        'n_ens': 5,

    }

    model = MLPEnsemble(
        config['layers'], n_models=config['n_ens'],
        activation=elu,
        reduction='mean')
    model.load_state_dict(state_dict)
    return model



if __name__  == '__main__':
    main()