# force floating point division. Can still use integer with //
from __future__ import division
# This file is used for importing the common utilities classes.
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.optimize import brute
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

class fit:
    def __init__(self,func_fit,func_predict,fit_dict,fit_result,fixed_kwargs):
        self.func_fit = func_fit,
        self.func_predict = func_predict
        self.fit_dict = fit_dict
        self.fit_result = fit_result
        self.fixed_kwargs = fixed_kwargs
    def predict(self,x):
        return self.func_predict(x,*(self.fit_result),**self.fixed_kwargs)

class LocalMinimization(object):
    def __init__(self,local_res,x,y,f,p0,y_err,kwargs):
        """
        :param local_res: result from curve_fit optimization
        :param x: see local_minimization
        :param y: see local_minimization
        :param f: see local_minimization
        :param p0: see local_minimization
        :param y_err: see local_minimization
        :param kwargs: see local_minimization
        """
        self.local_res = local_res
        self.x = x
        self.y = y
        self.f = f
        self.p0 = p0
        self.y_err = y_err
        self.kwargs = kwargs
    @property
    def p_opt(self):
        return self.local_res[0]
    @property
    def p_error(self):
        return np.sqrt(np.diag(self.local_res[1]))

def local_minimization(f, x_data, y_data, p0, y_err=None,absolute_sigma=None,
                       **kwargs):
    """
    :param f: function to call; called liked (x_data, *p0)
    :param x_data:  independent values
    :param y_data: dependent values
    :param p0: initial guess for params
    :param y_err: error in the y points
    :param kwargs: passed to curve_fit
    :return:
    """
    # see:
    # stackoverflow.com/questions/14581358/
    # getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    if y_err is not None and absolute_sigma is None:
        absolute_sigma = True
    local_res = curve_fit(f=f, xdata=x_data, ydata=y_data, p0=p0,
                          sigma=y_err, absolute_sigma=absolute_sigma, **kwargs)
    return LocalMinimization(local_res=local_res,x=x_data,y=y_data,
                             f=f,p0=p0,y_err=y_err,kwargs=kwargs)


def _l2(predicted,true):
    finite_pred = np.isfinite(predicted)
    finite_true = np.isfinite(true)
    valid_idx = np.where(finite_pred & finite_true)
    assert np.sum(finite_true) > 0 , "Brute giving completely non-finite objective"
    # POST: have at least one value; use the squared sum of all of them as
    # the penalty for infinite values
    worst_penalty = sum(np.abs(true[np.where(finite_true)])**2)
    values = np.ones(true.shape) * worst_penalty
    # where we are value, determine the actual penalty
    values[valid_idx] = \
        np.abs(predicted[valid_idx]-true[valid_idx])**2
    to_ret =  np.sum(values)
    return to_ret

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
    obj = _l2(predicted_values, true_values)
    return obj

def _grid_to_data(x,x_grid,y_grid,bounds_error=False):
    if (x_grid.size * y_grid.size > 0):
        # inteprolate from the (noisy) data to the (smooth) grid
        interpolator = interp1d(x=x_grid,y=y_grid,kind='linear',
                                fill_value='extrapolate',
                                bounds_error=bounds_error)
        predicted_values = interpolator(x)
    else:
        # we didn't find any valid extensions using these parameters; don't
        # use this model; give it an infinite loss.
        predicted_values = np.inf * np.ones(x.size)
    return predicted_values

def _l2_grid_to_data(ext,force,ext_grid,force_grid,**kw):
    """
    Gets the l2 loss associated between a model (gridded data) and actual
    data

    :param ext: the x values of the data
    :param force:  the y values of the data
    :param ext_grid: the x values of the model
    :param force_grid: the y values of the model
    :param **kw: see _grid_to_data
    :return: _l2 loss of the model onto the data
    """
    predicted_values = _grid_to_data(ext,ext_grid,force_grid,**kw)
    to_ret = _l2(predicted=predicted_values,true=force)
    return to_ret

def _prh_brute(objective,disp=False,full_output=False,**kwargs):
    """
    :param objective: function taking in arguments (must match ranges and Ns in
    **kwargs, see brute) and returning a loss
    :param disp: see brute
    :param full_output:  see: brute
    :param kwargs:  see : brute. Must specify ranges and Ns.
    :return:
    """
    assert "ranges" in kwargs , "Must specify ranges for brute"
    assert "Ns" in kwargs , "Must specify Ns for brute"
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
    objective = lambda *args: loss(func_to_call,true_values,*args)
    return _prh_brute(objective,**brute_dict)

def brute_fit(func_to_call,true_values,func_predict=None,fixed_kwargs=dict(),
              fit_dict=dict(),loss=objective_l2):
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
    brute_result = brute_optimize(func_to_call,true_values,brute_dict=fit_dict,
                                  loss=loss)
    return fit(func_fit=func_to_call,
               func_predict=func_predict,
               fit_dict=fit_dict,fixed_kwargs=fixed_kwargs,
               fit_result=brute_result)

def brute_then_bounded_minimize(f,x,y,ranges,yerr):
    """
    :param f: function, called like : f(x,r1,r2,..) where r_i corresponds
    to variable bounded by ranges[i]
    :param x:  idependent values
    :param y:  depenedent values
    :param ranges:  slices for each of the variable to fit
    :param yerr:for each of y 
    :return:
    """
    m_func = lambda *args: f(x,*args)
    Ns = [ int(np.ceil((r.stop-r.start)/r.step)) for r in ranges]
    fit_dict = dict(ranges=ranges, Ns=Ns, finish=None)
    fit = brute_fit(func_to_call=m_func,true_values=y,
                             fit_dict=fit_dict)
    # use the brute force method to force a local minimization
    half_steps = [r.step / 2 for r in ranges]
    bounds = [[p0 - h, p0 + h] for h, p0 in zip(half_steps, fit.fit_result)]
    bounds_lower = [min(b) for b in bounds]
    bounds_upper = [max(b) for b in bounds]
    bounds_local = (bounds_lower, bounds_upper)
    # minimize the l2 loss for local minimization
    f_local = lambda _F, *args: f(_F,*args)
    # y_data is zero, since ideally we have zero loss..
    kw_min = dict(f=f_local, x_data=x, y_data=y,
                  bounds=bounds_local, p0=fit.fit_result,
                  y_err=yerr,check_finite=True)
    fit_local = local_minimization(**kw_min)
    return fit_local