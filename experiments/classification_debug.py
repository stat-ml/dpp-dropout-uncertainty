import os
import pickle
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import entropy
from alpaca.uncertainty_estimator import build_estimator
from alpaca.uncertainty_estimator.masks import DEFAULT_MASKS

from configs import base_config, experiment_config
from classification_active_learning import loader


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    name = parser.parse_args().name

    config = deepcopy(base_config)
    config.update(experiment_config[name])
    config['name'] = name

    return config


def train(config, loaders, logdir, checkpoint=None):
    model = config['model_class']().double()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.eval()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(config['patience'])]

        runner = SupervisedRunner()
        runner.train(
            model, criterion, optimizer, loaders,
            logdir=logdir, num_epochs=config['epochs'], verbose=False,
            callbacks=callbacks
        )

    return model


def bench_uncertainty(model, model_checkpoint, loaders, x_val, y_val):
    runner = SupervisedRunner()
    logits = runner.predict_loader(model, loaders['valid'])
    probabilities = softmax(logits, axis=-1)

    estimators = ['max_entropy', 'max_prob', *DEFAULT_MASKS]

    uncertainties = {}
    for estimator_name in estimators:
        print(estimator_name)
        ue = calc_ue(model, x_val, probabilities, estimator_name, nn_runs=150)
        uncertainties[estimator_name] = ue

    record = {
        'checkpoint': model_checkpoint,
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'logits': logits,
        'estimators': estimators
    }
    with open(logdir / 'ue.pickle', 'wb') as f:
        pickle.dump(record, f)

    return probabilities, uncertainties, estimators


def misclassfication_detection(y_val, probabilities, uncertainties, estimators):
    results = []
    predictions = np.argmax(probabilities, axis=-1)
    errors = (predictions != y_val).astype(np.int)
    for estimator_name in estimators:
        fpr, tpr, _ = roc_curve(errors, uncertainties[estimator_name])
        roc_auc = roc_auc_score(errors, uncertainties[estimator_name])
        results.append((estimator_name, roc_auc))

    return results


def get_data(config):
    x_train, y_train, x_val, y_val, train_tfms = config['prepare_dataset'](config)

    if len(x_train) > config['train_size']:
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=config['train_size'], stratify=y_train
        )

    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size'], tfms=train_tfms, train=True),
        'valid': loader(x_val, y_val, config['batch_size'])
    })
    return loaders, x_train, y_train, x_val, y_val


def calc_ue(model, datapoints, probabilities, estimator_type='max_prob', nn_runs=150):
    if estimator_type == 'max_prob':
        ue = 1 - probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=-1)]
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
    else:
        estimator = build_estimator(
            'bald_masked', model, dropout_mask=estimator_type, num_classes=10,
            nn_runs=nn_runs, keep_runs=True, acquisition='var_ratio')
        ue = estimator.estimate(torch.DoubleTensor(datapoints).cuda())
    return ue


# # Set initial datas
# def loader(x, y, batch_size=128, shuffle=False):
#     ds = TensorDataset(torch.DoubleTensor(x), torch.LongTensor(y))
#     _loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=shuffle)
#     return _loader


if __name__ == '__main__':
    config = parse_arguments()
    loaders, x_train, y_train, x_val, y_val = get_data(config)

    rocaucs = []
    for i in range(config['repeats']):
        set_global_seed(i + 42)
        logdir = Path(f"logs/ht/{config['name']}_{i}")
        print(logdir)

        possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
        if os.path.exists(possible_checkpoint):
            checkpoint = possible_checkpoint
        else:
            checkpoint = None

        model = train(config, loaders, logdir, checkpoint)
        x_val_tensor = torch.cat([batch[0] for batch in loaders['valid']])

        probabilities, uncertainties, estimators = bench_uncertainty(
            model, checkpoint, loaders, x_val_tensor, y_val)

        aucs = misclassfication_detection(y_val, probabilities, uncertainties, estimators)
        rocaucs.extend(aucs)

    df = pd.DataFrame(rocaucs, columns=['Estimator', 'ROC-AUCs'])
    df.to_csv(f"logs/{config['name']}_ed.csv")
    plt.figure(figsize=(9, 6))
    plt.title(f"Error detection for {config['name']}")
    sns.boxplot('Estimator', 'ROC-AUCs', data=df)
    plt.savefig(f"data/ed/{config['name']}.png", dpi=150)
    plt.show()

