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

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette("muted")

from utils import *
from model import *

if __name__ == '__main__':
    
    TREAT_CTL_PATH = './data/seatac_vms1_vms2_treatment_ctl.csv'  # Data file
    treatment_control_df = pd.read_csv(TREAT_CTL_PATH, index_col='timestamp', parse_dates=True)
    
    ## Conduct regressions to get models and traces
    
    # First order
    ddt_delta_flow_model, ddt_delta_flow_trace = ddt_delta_flow_on_deparr_unnorm(
        treatment_ctl_df=treatment_control_df,
        nsamples=5000,
        prior_adherence_frac=0.01
    )
    with ddt_delta_flow_model:
        with open('./data/ddt_delta_flow_unnorm_model_trace.pkl', 'wb') as f:
            pickle.dump((ddt_delta_flow_model, ddt_delta_flow_trace), f)
    
    # Second-order
    ddt2_delta_flow_model, ddt2_delta_flow_trace = ddt_ddt_delta_flow_on_deparr_norm(
        treatment_ctl_df=treatment_control_df,
        nsamples=5000
    )
    with ddt2_delta_flow_model:
        with open('./data/ddt2_delta_flow_norm_model_trace.pkl', 'wb') as f:
            pickle.dump((ddt2_delta_flow_model, ddt2_delta_flow_trace), f)
    
    ## Plot combined treatment gaussians one-by-one
    
    # Plot constants
    SIDE = (20, 10)
    WIDTH=5
    FONTSIZE = 36
    OUTFILEPREF = './data/first_second_order_delflow'
    
    # Auto-regressive first
    ax = plot_combined_treatment_lagged_coefficients(
        ddt_delta_flow_trace, 'ddt_del_flow', SIDE, FONTSIZE, 'first-order', WIDTH, 'black'
    )
    ax = plot_combined_treatment_lagged_coefficients(
        ddt2_delta_flow_trace, 'ddt2_del_flow', SIDE, FONTSIZE, 'second-order', WIDTH, 'black', ax=ax, style='dashdot',
    )
    plt.savefig(f'{OUTFILEPREF}_autoreg.png')
    plt.cla()
    
    # Treat Departure
    ax = plot_combined_treatment_lagged_coefficients(
        ddt_delta_flow_trace, 'treatdep', SIDE, FONTSIZE, 'first-order', WIDTH, 'blue'
    )
    ax = plot_combined_treatment_lagged_coefficients(
        ddt2_delta_flow_trace, 'treatdep', SIDE, FONTSIZE, 'second-order', WIDTH, 'blue', ax=ax, style='dashdot',
    )
    plt.savefig(f'{OUTFILEPREF}_treatdep.png')
    plt.close()
    
    # Treat Arrival
    ax = plot_combined_treatment_lagged_coefficients(
        ddt_delta_flow_trace, 'treatarr', SIDE, FONTSIZE, 'first-order', WIDTH, 'orange'
    )
    ax = plot_combined_treatment_lagged_coefficients(
        ddt2_delta_flow_trace, 'treatarr', SIDE, FONTSIZE, 'second-order', WIDTH, 'orange', ax=ax, style='dashdot',
    )
    plt.savefig(f'{OUTFILEPREF}_treatarr.png')
    plt.close()