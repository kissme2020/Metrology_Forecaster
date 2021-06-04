# -*- coding: utf-8 -*-
"""
Created on <2021-06-01 Tue>.

python scripts for Fluke 732A 10V Ref Object and relate function

Fluke 734A csv data file format
model
S/N
Cal_Date,Measure,Uncr,Uncrr_Unit,k

@author: tskdkim
"""


# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
# import sys
# from pathlib import Path

# from datetime import datetime
# import importlib
# import numpy as np
# import matplotlib

# import csv
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Output, Input
# import plotly.express as px
# import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Own modules
from files import get_data_path, read_csv
# from myFile import DF_csv
# from myTime import My_timeDate
# from fund_class import Fund
# from index_class import Idx

# Package
from scipy.optimize import curve_fit
from scipy import stats
import statsmodels.api as sm

# from {path} import {class}


"""
Help function
"""


def read_data(fn):
    """Read Cal Data from csv file.

    fn: String file name with full path
    row1: model
    row2: S/N
    row3: columns
    below row 3, calibration data corresponding
    columns on row3
    Return model, sn, list of columns and Data Frame
    """
    # Read csv file
    alist = read_csv(fn)
    # Information of unit, Model, S/N, Unit
    infor = alist.pop(0)
    cols = alist.pop(0)

    # Get index of columns to Convert float
    # and Date
    num_cols_inds = [cols.index("Nominal"),
                     cols.index("Measure"),
                     cols.index("Uncer_95")]

    for vals in alist:
        # Iterate data list and converter it
        # as proper data type
        for i in range(len(vals)):
            # Iterate list of each row
            if i in num_cols_inds:
                # Convert it as float
                vals[i] = float(vals[i])

    # Create Data Frame for data
    data = pd.DataFrame(alist,
                        columns=cols)
    # Convert Cal_Data to time_date,
    data["Cal_Date"] = pd.to_datetime(data["Cal_Date"],
                                      format='%d-%b-%Y')
    # Return Model, S/N, columns, and Calibration Data
    return infor, data


"""
Function for linear approximation
"""


def first_order_func(x, a, b):
    """For linear Approximation.

    x: List or array, independence variable
    a: Float, coefficient of first order x
    b: Float, intercept of linear approximation
    """
    return a*x + b


"""
Class of Fluke 732A 10V REF
"""


# Test Code


"""
Read csv file
"""

# Get data file full path and file name
f_name = 'Fluke_732A70001.csv'
data_dir = 'data'
fn = get_data_path(os.getcwd(),
                   data_dir,
                   f_name)

# Read CSV file of Fluke_732A34567.csv
infor, data = read_data(fn)

# difference from nominal
y = data.Measure - data.Nominal


"""
Use Best fit of scipy.
Calculated Best Fit
"""


# Get difference as days of calibration date
cal_days = [0]  # Set initial daty as 0
# cal_days = []  # Set initial daty as 0
for i in range(len(data.Cal_Date) - 1):
    # Iterate Cal Date
    delta = data.Cal_Date[i+1] - data.Cal_Date[i]
    # cal_days.append(delta.days)
    cal_days.append(cal_days[-1] + delta.days)

cal_days = np.array(cal_days)

# Get linear approximation line
popt, pcov = curve_fit(first_order_func,
                       cal_days,
                       y)
# Linear fit
linear_fit_y = first_order_func(cal_days, *popt)


"""
Get prediction interval
y_0 +- t_crit * se

y_0: predicted value from linear approximation
t_crit: Inverse of the Student's t-distribution
se: standard error of the prediction
    s_yx * sqrt(1 + 1/n + ((x-x_mean)^2/ sample mean of x))
"""


# Get s_yx
x_prime = sm.add_constant(cal_days)
model = sm.OLS(y, x_prime)
fit = model.fit()
s_yx = np.sqrt(fit.mse_resid)
ss_x = ((cal_days - np.mean(cal_days)) ** 2.0).sum()
p_se = s_yx * np.sqrt(1 + (
    1 / len(y)) + (
    (cal_days - (np.mean(cal_days)) ** 2.0) / ss_x))
t_crit = stats.t.ppf(1 - 0.025,
                     len(cal_days)-2)

predict_up_err = linear_fit_y + (t_crit * p_se)
predict_low_err = linear_fit_y - (t_crit * p_se)

# Plot scatter for different from nominal value
mrk_size = 10
plt.scatter(data['Cal_Date'],
            y,
            color='b',
            marker='*',
            s=mrk_size)
# Plot error bar
plt.errorbar(data['Cal_Date'],
             y,
             yerr=data["Uncer_95"],
             linestyle="None",
             marker="None",
             color="b",
             # Change error bar's cap size
             capsize=mrk_size/5,
             # Change Error bar's thickness
             elinewidth=mrk_size/15)

# # Horizontal Line of Zero
# plt.axhline(y=0.0,
#             color='black',
#             linestyle='--',
#             linewidth=mrk_size/10,
#             label="Nomial")

# Plot linear approximation
plt.plot(data['Cal_Date'],
         linear_fit_y,
         linestyle='--',
         linewidth=mrk_size/9,
         label="best_fit_scipy")

# Plot linear approximation
plt.plot(data['Cal_Date'],
         predict_low_err,
         linestyle='--',
         linewidth=mrk_size/13,
         label="lower")

plt.plot(data['Cal_Date'],
         predict_up_err,
         linestyle='--',
         linewidth=mrk_size/13,
         label="upper")

plt.legend()
plt.show()
