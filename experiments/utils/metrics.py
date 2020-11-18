import numpy as np


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