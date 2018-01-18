# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import brute

class fit:
    def __init__(self,func_fit,func_predict,fit_dict,fit_result,fixed_kwargs):
        self.func_fit = func_fit,
        self.func_predict = func_predict
        self.fit_dict = fit_dict
        self.fit_result = fit_result
        self.fixed_kwargs = fixed_kwargs
    def predict(self,x):
        return self.func_predict(x,*(self.fit_result),**self.fixed_kwargs)
    
def objective_l2(func_predict,true_values,*args,**kwargs):
    """
    Returns the L2 (least squares) fit of the predicted values to the true,
    normalizd by (true_values)**2

    Args:
        func_predict: should take *args,**kwargs, return a list like true_values
        true_values: ground truth
        *args,**kwargs: for func_predict
    Returns:
        normalized L2 
    """
    # brute gives the fitting arguments like
    # tuple(array(arg1,arg2,...),)
    # so the first element is the (possibly single-element) array that
    # we want to pass as a *list*. may be a more elegant way of doing this 
    try:
        args = list(args[0])
    except TypeError:
        args = list([args[0]])
    predicted_values = func_predict(*args,**kwargs)
    finite_pred = np.isfinite(predicted_values)
    finite_true = np.isfinite(true_values)
    valid_idx = np.where(finite_pred & finite_true)
    assert sum(finite_true) > 0 , "Brute giving completely non-finite objective"
    # POST: have at least one value; use the squared sum of all of them as
    # the penalty for infinite values
    worst_penalty = sum(np.abs(true_values[np.where(finite_true)])**2)
    values = np.ones(true_values.size) * worst_penalty
    # where we are value, determine the actual penalty
    values[valid_idx] = \
        np.abs(predicted_values[valid_idx]-true_values[valid_idx])**2
    to_ret =  sum(np.log(values))
    return to_ret
    
def _prh_brute(objective,disp=False,full_output=False,**kwargs):
    return brute(objective,disp=disp,full_output=full_output,**kwargs)
    
def brute_optimize(func_to_call,true_values,loss=objective_l2,
                   brute_dict=dict()):
    """
    given a function to call, gets the brute-optimized parameter values
    
    Args:
        func_to_call: what to call, should just take in the fit parameters
        true_values: what the true results should be
        brute_dict: options for scipy.optimize.brute
    Returns:
        output of scipy.optimize
    """
    objective = lambda *args: objective_l2(func_to_call,true_values,*args)
    return _prh_brute(objective,**brute_dict)

def brute_fit(func_to_call,true_values,func_predict=None,fixed_kwargs=dict(),
              fit_dict=dict()):
    """
    given a function for fiting and a function for predicting, calls 
    brute_optimize and returns a fit object
    
    Args:
        func_to_call: what to call, should just take in the fit parameters
        true_values: what the true results should be
        **kwargs: see brute_optimize
    Returns:
        output of brute_optimize, wrapped to a fit object
    """
    brute_result = brute_optimize(func_to_call,true_values,brute_dict=fit_dict)
    return fit(func_fit=func_to_call,
               func_predict=func_predict,
               fit_dict=fit_dict,fixed_kwargs=fixed_kwargs,
               fit_result=brute_result)
