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


def _adaptor_function(f,list_elements,args,kwargs):
    """
    :param f: function taking in list, then *args and **kwargs
    :param list_elements: list-like
    :param args: arguments
    :param kwargs:  keyword arguments
    :return: whatever f returns
    """
    return f(list_elements,*args,**kwargs)

def _multiprocess_adapt(a):
    """
    :param a: assumed to be like <f, list_elements, args, kwargs>
    :return: see _adaptor_function
    """
    return _adaptor_function(*a)

def _map_helper(samples,n_cpus=1):
    """
    :param samples:  see _multiprocess_adapt (list, each element like
    <f, list_elements, args, kwargs>)
    :param n_cpus:
    :return:
    """
    if n_cpus > 1:
        Pool = multiprocessing.Pool(n_cpus)
        f_map = Pool.map
    else:
        f_map = map
    to_ret = f_map(_multiprocess_adapt,samples)
    return list(to_ret)

def bootstrap(samples,function,n_trials,n_cpus=1,seed=None,args=[],
              do_not_resample=False,**kwargs):
    """
    :param samples: list-like of length n
    :param function:  takes in list, then possible *args and ***kwargs,
    returns whatever we want. for multiprocessing, should be pickleable
    :param n_trials:  number of bootstrapping trials to implement
    :param n_cpus: number of CPUs to use
    :param seed: if not None, seeds the PRNG
    :param args: list, passed directly to function
    :param kwargs: dictionary, passed directly to function
    :param do_not_resample: if true, uses samples. Should only be used
    when n trials = 1 (easy way to get the 'normal', unsamples value)
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
    if do_not_resample:
        assert n_trials == 1 , "Randomization only disabled when using 1 trial"
        sample_ensembles = [ [function,samples,args,kwargs] ]
    else:
        idx = np.arange(n,dtype=np.int64)
        sample_ensembles = [ [function,
                              [samples[i] for i in np.random.choice(idx,size=n,replace=True)],
                              args,kwargs]
                             for _ in range(n_trials)]
    return _map_helper(sample_ensembles,n_cpus=n_cpus)

def bootstrap_with_standard(*args,**kwargs):
    kw_standard = dict(**kwargs)
    kw_standard["do_not_resample"] = True
    kw_standard["n_cpus"] = 1
    kw_standard["n_trials"] = 1
    normal = bootstrap(*args, **kw_standard)
    others = bootstrap(*args,**kwargs)
    return normal, others

def max_cpus():
    """
    :return: maximum number of cpus to use (one less than exists)
    """
    return multiprocessing.cpu_count() - 1
