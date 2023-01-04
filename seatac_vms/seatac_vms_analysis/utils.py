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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set()

def add_hr_of_day_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Adds a column to dataframe for hour of day, to allow easy splitting.

    Args:
        df (pd.DataFrame): The input dataframe

    Returns:
        pd.DataFrame: The dataframe with an hour-of-day column added as HoD
    """

    df['HoD'] = df.index.hour
    return df

def plot_combined_treatment_lagged_coefficients(
    trace_samples: np.ndarray,
    key: str,
    figsize: Tuple[int, int],
    fontsize: int,
    label:str,
    width: int=1,
    color: str='blue',
    ax: plt.Axes=None,
    style='-'
) -> plt.Axes:
    """_summary_

    Args:
        trace_samples (np.ndarray): Samples from the pymc for t and t-1 treatment coefficients
        key (str): Name of key used to lookup coefficient in pymc
        figsize (Tuple[int, int]): Size of figure
        fontsize (int): Font size
        label (str): Plot label
        width (int, optional): Line width. Defaults to 1.
        color (str, optional): Line color. Defaults to 'blue'.
        ax (plt.Axes, optional): Pyplot axis instance. Defaults to None.
        style (str, optional): Line style. Defaults to '-'.

    Returns:
        plt.Axes: _description_
    """
    
    # Create combined co-efficient with sum of means and l2 norm of stds of t/t-1 coefficients
    mu = np.abs(np.sum(trace_samples[key].mean(axis=0)))
    sigma = np.linalg.norm(trace_samples[key].std(axis=0))
    
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    pdf = stats.norm.pdf(x, mu, sigma)
    ax = sns.lineplot(x=x, y=pdf, ax=ax, color=color, linestyle = style, label=label, linewidth=width)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize)
    plt.legend(fontsize=fontsize)
    
    # Fill between 95% CIs
    lo = mu - 1.96*sigma
    hi = mu + 1.96*sigma
    
    ax.fill_between(x, pdf, where=(lo < x) & (x < hi), alpha=0.3, color=color)
    
    return ax