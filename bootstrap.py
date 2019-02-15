# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import brute
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import multiprocessing

def bootstrap(samples,function,n_trials,n_cpus=1,seed=None):
    """
    :param samples: list-like of length n
    :param function:  takes in list, returns whatever we want
    :param n_trials:  number of bootstrapping trials to implement
    :param n_cpus: number of CPUs to use
    :param seed: if not None, seeds the PRNG
    :return: list of length n_trials, each element is function applied
    to samples using randomly drawn samples
    """
    try:
        n = len(samples)
    except TypeError as e:
        print(e)
        assert False , "Bootstrap must be passed a list-like"
        return
    if seed is not None:
        np.random.seed(seed)
    sample_ensembles = [ np.random.choice(samples,size=n,replace=True)
                         for _ in range(n_trials)]
    if n_cpus > 1:
        Pool = multiprocessing.Pool(n_cpus)
        f_map = Pool.map
    else:
        f_map = map
    to_ret = f_map(function,sample_ensembles)
    return to_ret
