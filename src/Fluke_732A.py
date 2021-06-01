# -*- coding: utf-8 -*-
"""
Created on <2021-06-01 Tue>

python scripts for Fluke 732A 10V Ref Object and relate function

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
# import matplotlib.pyplot as plt
# import pandas as pd
# import csv
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Output, Input
# import plotly.express as px
# import plotly.graph_objects as go

# Own modules
from files import get_data_path, read_csv
# from myFile import DF_csv
# from myTime import My_timeDate
# from fund_class import Fund
# from index_class import Idx

# libs

# from {path} import {class}


"""
Class of Fluke 732A 10V REF
"""



# Test Code

"""
Read csv file
"""

# Get data file full path and file name
f_name = 'Fluke_732A34567.csv'
data_dir = 'data'
fn = get_data_path(os.getcwd(),
                   data_dir,
                   f_name)

# Read CSV file of Fluke_732A34567.csv
data = read_csv(fn)

for val in data:
    print(val)
