import os
import random

import numpy as np
import torch

from alpaca.uncertainty_estimator.nngp import NNGPRegression
from alpaca.uncertainty_estimator.mcdue import MCDUE, MCDUEMasked
from alpaca.uncertainty_estimator.eue import EnsembleUE, EnsembleMCDUE, EnsembleNLLUE, EnsembleMCDUEMasked
from alpaca.uncertainty_estimator.random_estimator import RandomEstimator
from alpaca.uncertainty_estimator.bald import Bald, BaldMasked, BaldEnsemble


def build_estimator(name, model, **kwargs):
    if name == 'nngp':
        estimator = NNGPRegression(model, **kwargs)
    elif name == 'random':
        estimator = RandomEstimator()
    elif name == 'mcdue':
        estimator = MCDUE(model, **kwargs)
    elif name == 'mcdue_masked':
        estimator = MCDUEMasked(model, **kwargs)
    elif name == 'bald':
        estimator = Bald(model, **kwargs)
    elif name == 'bald_masked':
        estimator = BaldMasked(model, **kwargs)
    elif name == 'bald_ensemble':
        estimator = BaldEnsemble(model, **kwargs)
    elif name == 'eue_nll':
        estimator = EnsembleNLLUE(model)
    elif name == 'eue':
        estimator = EnsembleUE(model)
    elif name == 'emcdue':
        estimator = EnsembleMCDUE(model, **kwargs)
    elif name == 'emcdue_masked':
        estimator = EnsembleMCDUEMasked(model, **kwargs)
    else:
        raise ValueError("Wrong estimator name")
    return estimator


def set_random(random_seed):
    # Setting seeds for reproducibility
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def get_model(model, model_path, train_set=None, val_set=None, retrain=False, **kwargs):
    model_path = os.path.join(ROOT_DIR, model_path)
    if retrain or not os.path.exists(model_path):
        if train_set is None or val_set is None:
            raise RuntimeError("You should pass datasets for retrain")
        model.fit(train_set, val_set, **kwargs)
        torch.save(model.state_dict(), model_path)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
    return model
