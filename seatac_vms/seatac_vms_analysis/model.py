"""
MIT License

Copyright (c) 2022 Shushman Choudhury

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn import preprocessing

def standardize_series(
    series: pd.Series
) -> np.ndarray:
    """Standardize by subtracting mean from all values and dividing by stdev.

    Args:
        series (pd.Series): Input series

    Returns:
        np.ndarray: Standardized numpy array
    """
    series_np = series.to_numpy().reshape(-1, 1)
    series_np_norm = preprocessing.StandardScaler().fit(series_np).transform(series_np)
    
    return series_np_norm

def ddt_delta_flow_on_deparr_unnorm(
    treatment_ctl_df: pd.DataFrame,
    nsamples: int,
    prior_adherence_frac:float
) -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
    """Compute the regression on the unstandardized first-order time derivative of difference in flow, as per paper.

    Args:
        treatment_ctl_df (pd.DataFrame): Overall treatment-control dataframe
        nsamples (int): Number of samples for which to do Hamiltonian Monte Carlo.
        prior_adherence_frac (float): Prior on the proportion of traffic that adheres to sign.

    Returns:
        Tuple[pm.Model, pm.backends.base.MultiTrace]: Model and Trace.
    """
    
    treatment_ctl_df['treat_dep_lagged_1'] = treatment_ctl_df['treat_dep'].shift(1)
    treatment_ctl_df['treat_arr_lagged_1'] = treatment_ctl_df['treat_arr'].shift(1)

    # Add delta flow column
    treatment_ctl_df['delta_flow'] = treatment_ctl_df.dep_flow - treatment_ctl_df.arr_flow
    
    # Discrete-time gradient of delta flow
    treatment_ctl_df['ddt_delta_flow'] = treatment_ctl_df['delta_flow'] - treatment_ctl_df['delta_flow'].shift(1)
    treatment_ctl_df['ddt_delta_flow_lagged_1'] = treatment_ctl_df['ddt_delta_flow'].shift(1)

    
    # At end, drop NaT rows
    treatment_ctl_df.dropna(inplace=True)

    # Don't standardize data
    ddt_delta_flow = treatment_ctl_df.ddt_delta_flow.to_numpy().reshape(-1, 1)
    ddt_delta_flow_lagged1 = treatment_ctl_df.ddt_delta_flow_lagged_1.to_numpy().reshape(-1, 1)
    
    # Find neg and pos chunks of ddt_delta_flow (for coefficients on treatment)
    ddt_delta_flow_pos = ddt_delta_flow[np.where(ddt_delta_flow > 0.0)[0]]
    ddt_delta_flow_neg = ddt_delta_flow[np.where(ddt_delta_flow < 0.0)[0]]
    
    # Get reshaped -1,1
    treat_dep = treatment_ctl_df.treat_dep.to_numpy().reshape(-1, 1)
    treat_dep_lagged_1 = np.multiply(treat_dep, treatment_ctl_df.treat_dep_lagged_1.to_numpy().reshape(-1, 1))
    treat_arr = treatment_ctl_df.treat_arr.to_numpy().reshape(-1, 1)
    treat_arr_lagged_1 = np.multiply(treat_arr, treatment_ctl_df.treat_arr_lagged_1.to_numpy().reshape(-1, 1))
    
      
    delta_flow_model = pm.Model()
    
    with delta_flow_model:
        
        sigma = pm.Exponential("sigma", 1.0/ddt_delta_flow.std())
        
        # Set mean to 0 because if past is 0, so will this be
        intercept = pm.Normal("intercept", 0.0, ddt_delta_flow.std())
        
        slope_delflow = pm.Normal("ddt_del_flow", 0.0, 1.0, shape=1)
        
        # Expect negative effect of treat dep
        slope_treatdep = pm.Normal("treatdep", prior_adherence_frac*ddt_delta_flow_neg.mean(),
                                   ddt_delta_flow_neg.std(), shape=2)
        
        # Expect positive effect of treat arr
        slope_treatarr = pm.Normal("treatarr", prior_adherence_frac*ddt_delta_flow_pos.mean(),
                                   ddt_delta_flow_pos.std(), shape=2)
        

        likelihood = pm.Normal(
            "y",
            mu=intercept + slope_delflow[0]*ddt_delta_flow_lagged1 +\
                slope_treatdep[0]*treat_dep + slope_treatdep[1]*treat_dep_lagged_1 +\
                slope_treatarr[0]*treat_arr + slope_treatarr[1]*treat_arr_lagged_1,
            sigma=sigma,
            observed=ddt_delta_flow    
        )
    
        trace = pm.sample(nsamples, return_inferencedata=False)
    
    return (delta_flow_model, trace)


def ddt_ddt_delta_flow_on_deparr_norm(
    treatment_ctl_df: pd.DataFrame,
    nsamples: int
) -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
    """Compute the regression on the standardized second-order time derivative of difference in flow, as per paper.

    Args:
        treatment_ctl_df (pd.DataFrame): Overall treatment-control dataframe
        nsamples (int): Number of samples for which to do Hamiltonian Monte Carlo.

    Returns:
        Tuple[pm.Model, pm.backends.base.MultiTrace]: Model and Trace.
    """
    
    treatment_ctl_df['treat_dep_lagged_1'] = treatment_ctl_df['treat_dep'].shift(1)
    treatment_ctl_df['treat_arr_lagged_1'] = treatment_ctl_df['treat_arr'].shift(1)

    # Add delta flow column
    treatment_ctl_df['delta_flow'] = treatment_ctl_df.dep_flow - treatment_ctl_df.arr_flow
    treatment_ctl_df['ddt_delta_flow'] = treatment_ctl_df['delta_flow'] - treatment_ctl_df['delta_flow'].shift(1)
    treatment_ctl_df['ddt_ddt_delta_flow'] = treatment_ctl_df['ddt_delta_flow'] - treatment_ctl_df['ddt_delta_flow'].shift(1)
    treatment_ctl_df['ddt_ddt_delta_flow_lagged_1'] = treatment_ctl_df['ddt_ddt_delta_flow'].shift(1)
    
    
    # At end, drop NaT rows
    treatment_ctl_df.dropna(inplace=True)

    # normalize data
    ddt_ddt_delta_flow_norm = standardize_series(treatment_ctl_df.ddt_ddt_delta_flow)
    ddt_ddt_delta_flow_lagged1_norm = standardize_series(treatment_ctl_df.ddt_ddt_delta_flow_lagged_1)
    
    # Get reshaped -1,1
    treat_dep = treatment_ctl_df.treat_dep.to_numpy().reshape(-1, 1)
    treat_dep_lagged_1 = np.multiply(treat_dep, treatment_ctl_df.treat_dep_lagged_1.to_numpy().reshape(-1, 1))
    treat_arr = treatment_ctl_df.treat_arr.to_numpy().reshape(-1, 1)
    treat_arr_lagged_1 = np.multiply(treat_arr, treatment_ctl_df.treat_arr_lagged_1.to_numpy().reshape(-1, 1))
    
        
    ddt_ddt_delta_flow_model = pm.Model()
    
    with ddt_ddt_delta_flow_model:
        
        sigma = pm.Exponential("sigma", 1.0)
        intercept = pm.Normal("intercept", 0.0, 1.0)
        
        slope_delflow = pm.Normal("ddt2_del_flow", 0.0, 1.0, shape=1)
        
        # Expect negative effect of treat dep
        slope_treatdep = pm.Normal("treatdep", -0.5, 1.0, shape=2)
        
        # Expect positive effect of treat arr
        slope_treatarr = pm.Normal("treatarr", 0.5, 1.0, shape=2)
        
        
        likelihood = pm.Normal(
            "y",
            mu=intercept + slope_delflow[0]*ddt_ddt_delta_flow_lagged1_norm +\
                slope_treatdep[0]*treat_dep + slope_treatdep[1]*treat_dep_lagged_1 +\
                slope_treatarr[0]*treat_arr + slope_treatarr[1]*treat_arr_lagged_1,
            sigma=sigma,
            observed=ddt_ddt_delta_flow_norm    
        )
    
        trace = pm.sample(nsamples, return_inferencedata=False)
    
    return (ddt_ddt_delta_flow_model, trace)