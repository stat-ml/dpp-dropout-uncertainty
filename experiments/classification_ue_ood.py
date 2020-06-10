import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from catalyst.dl import SupervisedRunner
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import entropy
from alpaca.uncertainty_estimator import build_estimator
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS

from configs import base_config, experiment_ood_config
from deprecated.classification_active_learning import loader

from classification_ue import train


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    args = parser.parse_args()

    config = deepcopy(base_config)
    config.update(experiment_ood_config[args.name])
    config['name'] = args.name
    config['acquisition'] = args.acquisition

    return config


def bench_uncertainty(model, model_checkpoint, ood_loader, x_ood, acquisition, nn_runs):
    runner = SupervisedRunner()
    logits = runner.predict_loader(model, ood_loader)
    probabilities = softmax(logits, axis=-1)

    if config['acquisition'] in ['bald', 'var_ratio']:
        estimators = ['mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    elif config['acquisition'] == 'max_prob':
        estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    else:
        raise ValueError


    uncertainties = {}
    for estimator_name in estimators:
        print(estimator_name)
        ue = calc_ue(model, x_ood, probabilities, estimator_name, nn_runs=nn_runs, acquisition=acquisition)
        uncertainties[estimator_name] = ue
        print(ue)

    record = {
        'checkpoint': model_checkpoint,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'logits': logits,
        'estimators': estimators
    }

    file_name = logdir / f"ue_ood_{config['acquisition']}.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(record, f)

    return probabilities, uncertainties, estimators


def calc_ue(model, datapoints, probabilities, estimator_type='max_prob', nn_runs=100, acquisition='bald'):
    if estimator_type == 'max_prob':
        ue = 1 - np.max(probabilities, axis=-1)
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        acquisition_param = 'var_ratio' if acquisition == 'max_prob' else acquisition

        estimator = build_estimator(
            'bald_masked', model, dropout_mask=estimator_type, num_classes=10,
            nn_runs=nn_runs, keep_runs=True, acquisition=acquisition_param)
        ue = estimator.estimate(torch.DoubleTensor(datapoints).cuda())

        if acquisition == 'max_prob':
            probs = softmax(estimator.last_mcd_runs(), axis=-1)
            probs = np.mean(probs, axis=-2)
            ue = 1 - np.max(probs, axis=-1)

    return ue


def get_data(config):
    x_train, y_train, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    if len(x_train) > config['train_size']:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=config['train_size'], stratify=y_train
        )

    _, _, x_ood, y_ood, _ = config['prepare_dataset'](config)
    ood_loader = loader(x_ood, y_ood, config['batch_size'])

    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size'], tfms=train_tfms, train=True),
        'valid': loader(x_val, y_val, config['batch_size']),
    })
    return loaders, ood_loader, x_train, y_train, x_val, y_val, x_ood, y_ood


if __name__ == '__main__':
    config = parse_arguments()
    print(config)
    set_global_seed(42)
    loaders, ood_loader, x_train, y_train, x_val, y_val, x_ood, y_ood = get_data(config)
    print(y_train[:5])

    for i in range(config['repeats']):
        set_global_seed(i + 42)
        logdir = Path(f"logs/classification/{config['name']}_{i}")
        print(logdir)

        possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
        if os.path.exists(possible_checkpoint):
            checkpoint = possible_checkpoint
        else:
            checkpoint = None

        model = train(config, loaders, logdir, checkpoint)
        print(model)
        x_ood_tensor = torch.cat([batch[0] for batch in ood_loader])

        probabilities, uncertainties, estimators = bench_uncertainty(
            model, checkpoint, ood_loader, x_ood_tensor, config['acquisition'], config['nn_runs'])

