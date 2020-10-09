from collections import defaultdict
import torch
import numpy as np
from dppy.finite_dpps import FiniteDPP

ATTEMPTS = 5


class BasicBernoulliMask:
    def __init__(self):
        self.dry_run = False

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        p = 1 - dropout_rate

        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))

        noise = torch.zeros(x.shape[-1]).double().to(x.device)
        # noise = self._make_noise(x)
        if p > 0:
            noise.bernoulli_(p).div_(p)
        # noise = noise.expand_as(x)
        return noise


class DPPMask:
    def __init__(self, ht_norm=False, covariance=False):
        self.dpps = {}
        self.layer_correlations = {}  # keep for debug purposes # Flag for uncertainty estimator to make first run without taking the result
        self.dry_run = True
        self.ht_norm = ht_norm
        self.norm = {}
        self.covariance = covariance

        ## For batch loaders
        self.freezed = False
        self.masks = {}

    def __call__(self, x, dropout_rate=0.5, layer_num=0):
        mask_len = x.shape[-1]
        if layer_num not in self.layer_correlations:
            # warm-up, generatign correlations masks
            x_matrix = x.cpu().numpy()
            if self.freezed:
                self.x_matrices[layer_num].append(x_matrix)
            else:
                self._setup_dpp(x_matrix, layer_num)
            return torch.ones(mask_len).to(x.device)

        if self.freezed:
            mask = self.masks[layer_num]
        else:
            mask = self._generate_mask(layer_num, mask_len)
        return mask

    def freeze(self, dry_run):
        self.freezed = True
        if dry_run:
            self.x_matrices = defaultdict(list)
        else:
            for layer_num in self.x_matrices.keys():
                mask_len = len(self.layer_correlations[layer_num])
                self.masks[layer_num] = self._generate_mask(layer_num, mask_len)

    def unfreeze(self, dry_run):
        self.freezed = False
        if dry_run:
            for layer_num, matrices in self.x_matrices.items():
                x_matrix = np.concatenate(matrices)
                self._setup_dpp(x_matrix, layer_num)

    def _setup_dpp(self, x_matrix, layer_num):
        self.x_matrix = x_matrix
        micro = 1e-12
        x_matrix += np.random.random(x_matrix.shape) * micro  # for computational stability
        if self.covariance:
            L = np.cov(x_matrix.T)
        else:
            L = np.corrcoef(x_matrix.T)

        self.dpps[layer_num] = FiniteDPP('likelihood', **{'L': L})
        self.layer_correlations[layer_num] = L

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.ht_norm:
            L = torch.DoubleTensor(L).to(device)
            I = torch.eye(len(L)).double().to(device)
            K = torch.mm(L, torch.inverse(L + I))

            self.norm[layer_num] = torch.reciprocal(torch.diag(K))  # / len(correlations)
            self.L = L
            self.K = K

    def _generate_mask(self, layer_num, mask_len):
        # sampling nodes ids
        dpp = self.dpps[layer_num]

        for _ in range(ATTEMPTS):
            dpp.sample_exact()
            ids = dpp.list_of_samples[-1]
            if len(ids):  # We should retry if mask is zero-length
                break

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask = torch.zeros(mask_len).double().to(device)
        if self.ht_norm:
            mask[ids] = self.norm[layer_num][ids]
        else:
            mask[ids] = mask_len / len(ids)

        return mask

    def reset(self):
        self.layer_correlations = {}


class MCDUEMasked:
    """
    Estimate uncertainty for samples with MCDUE approach
    """
    def __init__(self, net, nn_runs=25, dropout_rate=.5, dropout_mask=None, keep_runs=False):
        self.net = net
        self.nn_runs = nn_runs
        self.dropout_rate = dropout_rate
        # if isinstance(dropout_mask, str):
        #     dropout_mask = build_mask(dropout_mask)
        self.dropout_mask = dropout_mask
        self.keep_runs = keep_runs
        self._mcd_runs = np.array([])

    def estimate(self, X_pool, *args):
        mcd_runs = np.zeros((X_pool.shape[0], self.nn_runs))

        with torch.no_grad():
            # Some mask needs first run without dropout, i.e. decorrelation mask
            if hasattr(self.dropout_mask, 'dry_run') and self.dropout_mask.dry_run:
                self.net(X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask)

            # Get mcdue estimation
            for nn_run in range(self.nn_runs):
                print(nn_run)
                prediction = self.net(
                    X_pool, dropout_rate=self.dropout_rate, dropout_mask=self.dropout_mask
                ).to('cpu')
                mcd_runs[:, nn_run] = np.ravel(prediction)

            if self.keep_runs:
                self._mcd_runs = mcd_runs

        return np.ravel(np.std(mcd_runs, axis=1))

    def reset(self):
        if hasattr(self.dropout_mask, 'reset'):
            self.dropout_mask.reset()

    def last_mcd_runs(self):
        """Return model prediction for last uncertainty estimation"""
        if not self.keep_runs:
            print("mcd_runs: You should set `keep_runs=True` to properly use this method")
        return self._mcd_runs
