# -*- coding: utf-8 -*-
"""
Created on <2021-06-01 Tue>

python scripts to read write file.

@author: tskdkim
"""


# Futures
from __future__ import print_function

# Built-in/Generic Imports

# import sys
import os
from pathlib import Path

# from datetime import datetime
# import importlib
# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd
import csv
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Output, Input
# import plotly.express as px
# import plotly.graph_objects as go

# Own modules
# from myFile import DF_csv
# from myTime import My_timeDate
# from fund_class import Fund
# from index_class import Idx

# libs

# from {path} import {class}


"""
Path related function
"""


def get_data_path(path,
                  data_dir,
                  fn=None):
    """Return data path of given working path.

    path: String, path
    data_dir: String, directory name of data
    fn: String File name, Default None
    """
    # Get parent path
    p = Path(path)

    if fn is not None:
        # If file name Exits
        return os.path.join(p.parent.absolute(),
                            data_dir,
                            fn)
    else:
        # If file name not Exits
        return os.path.join(p.parent.absolute(),
                            data_dir)


def read_csv(fn, delimiterStr=','):
    """Read csv file.

    fn: file name
    return as a string list.
    """
    result = []  # s alist of result.
    with open(fn, encoding='utf-8-sig') as csv_file:
        csv_read = csv.reader(csv_file, delimiter=delimiterStr)
        for raw in csv_read:
            result.append(raw)

    return result
