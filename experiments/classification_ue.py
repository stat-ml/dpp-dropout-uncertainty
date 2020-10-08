import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.special import softmax
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, EarlyStoppingCallback
from catalyst.utils import set_global_seed

from alpaca.active_learning.simple_update import entropy
from alpaca.uncertainty_estimator import build_estimator

from model_resnet import ResNet34

from datasets import get_data


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--acquisition', '-a', type=str, default='bald')
    args = parser.parse_args()

    config = {}
    config['name'] = args.name
    config['repeats'] = 5
    config['nn_runs'] = 100
    config['acquisition'] = args.acquisition

    return config


def make_model():
    model = ResNet34()
    return model


def train(config, loaders, logdir, checkpoint=None):
    model = make_model()
    # model = config['model_class'](dropout_rate=config['dropout_rate']).double()

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        model.eval()
    else:
        runner = SupervisedRunner()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
        callbacks = [AccuracyCallback(num_classes=10), EarlyStoppingCallback(100, metric='accuracy01', minimize=False)]

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 160], gamma=0.1)
        model.train()
        runner.train(
            model=model, criterion=criterion, optimizer=optimizer, loaders=loaders,
            logdir=logdir, num_epochs=200, verbose=True, scheduler=scheduler,
            callbacks=callbacks
        )

    model.eval()

    return model


def bench_uncertainty(model, model_checkpoint, loaders, x_val, y_val, config):
    model = model.double().cuda()
    # runner = SupervisedRunner()
    # logits = runner.predict_loader(loader=loaders['valid'], model=model)
    # logits = torch.cat([l['logits'] for l in logits], dim=0).cpu()
    with torch.no_grad():
        logits = model(x_val.cuda()).detach().cpu().numpy()
    probabilities = softmax(logits, axis=-1)

    acquisition = 'max_prob'
    estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp']
    # if config['acquisition'] in ['bald', 'var_ratio']:
    #     estimators = ['mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    # elif config['acquisition'] == 'max_prob':
    #     estimators = ['max_prob', 'mc_dropout', 'ht_dpp', 'ht_k_dpp', 'cov_dpp', 'cov_k_dpp']
    # else:
    #     raise ValueError

    print(estimators)

    uncertainties = {}
    lls = {}
    mcds = {}
    for estimator_name in estimators:
        # try:
        print(estimator_name)
        ue, ll, mcd = calc_ue(model, x_val, y_val, probabilities, estimator_name, nn_runs=config['nn_runs'], acquisition=acquisition)
        uncertainties[estimator_name] = ue
        lls[estimator_name] = ll
        mcds[estimator_name] = mcd

        # except Exception as e:
        #     print(e)

    record = {
        'checkpoint': model_checkpoint,
        'y_val': y_val,
        'uncertainties': uncertainties,
        'probabilities': probabilities,
        'logits': logits,
        'estimators': estimators,
        'lls': lls,
        'mcd': mcds
    }
    with open(logdir / f"ue.pickle", 'wb') as f:
        pickle.dump(record, f)

    return probabilities, uncertainties, estimators


def log_likelihood(probabilities, y):
    try:
        ll = np.mean(np.log(probabilities[np.arange(len(probabilities)), y]))
    except FloatingPointError:
        import ipdb; ipdb.set_trace()
    return ll


def calc_ue(model, datapoints, y_val, probabilities, estimator_type='max_prob', nn_runs=100, acquisition='bald'):
    if estimator_type == 'max_prob':
        ue = 1 - np.max(probabilities, axis=-1)
        ll = log_likelihood(probabilities, y_val)
        mcd = probabilities
    elif estimator_type == 'max_entropy':
        ue = entropy(probabilities)
        ll = log_likelihood(probabilities, y_val)
        mcd = probabilities
    else:
        acquisition_param = 'var_ratio' if acquisition == 'max_prob' else acquisition

        estimator = build_estimator(
            'bald_masked', model, dropout_mask=estimator_type, num_classes=10,
            nn_runs=nn_runs, keep_runs=True, acquisition=acquisition_param)
        ue = estimator.estimate(torch.DoubleTensor(datapoints).cuda())
        probs = softmax(estimator.last_mcd_runs(), axis=-1)
        probs = np.mean(probs, axis=-2)
        ll = log_likelihood(probs, y_val)
        mcd = estimator.last_mcd_runs()

        if acquisition == 'max_prob':
            ue = 1 - np.max(probs, axis=-1)

    return ue, ll, mcd


def misclassfication_detection(y_val, probabilities, uncertainties, estimators):
    results = []
    predictions = np.argmax(probabilities, axis=-1)
    errors = (predictions != y_val)
    for estimator_name in estimators:
        fpr, tpr, _ = roc_curve(errors, uncertainties[estimator_name])
        roc_auc = roc_auc_score(errors, uncertainties[estimator_name])
        results.append((estimator_name, roc_auc))

    return results


if __name__ == '__main__':
    config = parse_arguments()
    set_global_seed(42)
    loaders = get_data(config['name'])

    rocaucs = []
    for i in range(config['repeats']):
        set_global_seed(i + 42)
        logdir = Path(f"logs/classification_better/{config['name']}_{i}")
        print(logdir)

        possible_checkpoint = logdir / 'checkpoints' / 'best.pth'
        if os.path.exists(possible_checkpoint):
            checkpoint = possible_checkpoint
        else:
            checkpoint = None

        model = train(config, loaders, logdir, checkpoint)

        x_val = []
        y_val = []
        for x, y in loaders['valid']:
            x_val.append(x)
            y_val.append(y)
        x_val = torch.cat(x_val).double()
        y_val = torch.cat(y_val).numpy()

        bench_uncertainty(
            model, checkpoint, loaders, x_val, y_val, config)
