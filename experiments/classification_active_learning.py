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

from configs import al_config, al_experiments


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    name = parser.parse_args().name

    config = deepcopy(al_config)
    config.update(al_experiments[name])
    config['name'] = name

    return config


def train(config, loaders, logdir):
    model = config['model_class']().double()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(config['patience'])]

    runner = SupervisedRunner()
    runner.train(
        model, criterion, optimizer, loaders,
        logdir=logdir, num_epochs=config['epochs'], verbose=False,
        callbacks=callbacks
    )


def active_train(config):
    # Load data
    x_set, y_set, x_val, y_val, train_tfms = config['prepare_dataset'](config)
    print(x_set.shape)

    return [0]

    val_accuracy = []
    for _ in range(config['repeats']):  # more repeats for robust results
        # Initial data split
        x_set, x_train_init, y_set, y_train_init = train_test_split(x_set, y_set, test_size=config['start_size'], stratify=y_set)
        _, x_pool_init, _, y_pool_init = train_test_split(x_set, y_set, test_size=config['pool_size'], stratify=y_set)

        # Active learning
        for method in config['methods']:
            print(f"== {method} ==")
            x_pool, y_pool = np.copy(x_pool_init), np.copy(y_pool_init)
            x_train, y_train = np.copy(x_train_init), np.copy(y_train_init)

            model = config['model_class']().double()
            accuracies = []

            for i in range(config['steps']):
                print(f"Step {i+1}, train size: {len(x_train)}")

                # learner = train_classifier(model, config, x_train, y_train, x_val, y_val, train_tfms)
                # accuracies.append(learner.recorder.metrics[-1][0].item())

                if i != config['steps'] - 1:
                    x_pool, x_train, y_pool, y_train = update_set(
                        x_pool, x_train, y_pool, y_train, config['step_size'], method=method, model=model)

            records = list(zip(accuracies, range(len(accuracies)), [method] * len(accuracies)))
            val_accuracy.extend(records)

    # Display results
    try:
        plot_metric(val_accuracy, config)
    except:
        import ipdb; ipdb.set_trace()




def bench_error_detection(model, estimators, loaders, x_val, y_val):
    runner = SupervisedRunner()
    logits = runner.predict_loader(model, loaders['valid'])
    probabilities = softmax(logits, axis=-1)

    estimators = [
        # 'max_entropy', 'max_prob',
        'mc_dropout', 'decorrelating_sc',
        'dpp', 'ht_dpp', 'k_dpp', 'ht_k_dpp']

    uncertainties = {}
    for estimator_name in estimators:
        print(estimator_name)
        ue = calc_ue(model, x_val, probabilities, estimator_name, nn_runs=150)
        uncertainties[estimator_name] = ue

    predictions = np.argmax(probabilities, axis=-1)
    errors = (predictions != y_val).astype(np.int)

    results = []
    for estimator_name in estimators:
        fpr, tpr, _ = roc_curve(errors, uncertainties[estimator_name])
        roc_auc = roc_auc_score(errors, uncertainties[estimator_name])
        results.append((estimator_name, roc_auc))

    return results


def get_loaders(x_train, y_train, x_val, y_val, config, tfms):
    loaders = OrderedDict({
        'train': loader(x_train, y_train, config['batch_size'], shuffle=True),
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


# Set initial datas
def loader(x, y, batch_size=128, shuffle=False):
    ds = TensorDataset(torch.DoubleTensor(x), torch.LongTensor(y))
    _loader = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=shuffle)
    return _loader


if __name__ == '__main__':
    config = parse_arguments()
    results = []
    for i in range(config['repeats']):
        accuracies = active_train(config)
        results.extend(accuracies)


    # rocaucs = []
    # for i in range(config['repeats']):
    #     set_global_seed(i + 42)
    #     logdir = f"logs/ht/{config['name']}_{i}"
    #     model = train(config, loaders, logdir)
    #     rocaucs.extend(bench_error_detection(model, config, loaders, x_val, y_val))
    #
    # df = pd.DataFrame(rocaucs, columns=['Estimator', 'ROC-AUCs'])
    # plt.figure(figsize=(9, 6))
    # plt.title(f"Error detection for {config['name']}")
    # sns.boxplot('Estimator', 'ROC-AUCs', data=df)
    # plt.savefig(f"data/al/{config['name']}.png", dpi=150)
    # plt.show()
    #
    # df.to_csv(f"logs/{config['name']}_al.csv")


