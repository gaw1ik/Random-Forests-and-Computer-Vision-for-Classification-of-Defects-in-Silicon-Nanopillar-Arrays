# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:11:05 2020

This code makes the dataframe Devices_Training
which a random sample of size N (set to 200)
from Devices_Dfct.

@author: Brian
"""

#%% IMPORT

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd

#%% LOAD

Devices_Dfct_Info = pd.read_pickle('Data\Devices_Dfct_Info')
Devices_Dfct_Features = pd.read_pickle('Data\Devices_Dfct_Features')

#%% Randomly sample n rows from Devices_Dfct to create Devices_Training

'''random state (RS) chosen as 42 for consistency 
(keep consistent between _INFO and _FEATURES
otherwise they won't contain corresponding devices)
'''

RS = 42
N = 200

Devices_Training_Info     = Devices_Dfct_Info.sample    (n=N, random_state=RS, axis=0)
Devices_Training_Features = Devices_Dfct_Features.sample(n=N, random_state=RS, axis=0)

#%% SAVE

Devices_Training_Info.to_pickle('Data\Devices_Training_Info')
Devices_Training_Features.to_pickle('Data\Devices_Training_Features')