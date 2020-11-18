import numpy as np
from scipy.special import logsumexp


def uq_ll(errors, uncertainty):
    """
    measures the log likelihood (not negative, the bigger - the better)

    :param errors: absolute values of difference between prediction and true values
    :param uncertainty: the std (square root of variance) is expected here
    :return: log likelihood, i.e. how well the uncertainty estimated for regression
    """

    variance = np.square(uncertainty)
    lls = -1/2 * (np.log(variance) + np.square(errors)/variance + np.log(np.pi))
    return np.mean(lls)


def tau_ll(tau, y, y_hat, T):
    ll = np.mean(
        logsumexp(-0.5 * tau * (y[None] - y_hat)**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau)
    )
    return ll
