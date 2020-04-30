# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:11:05 2020

@author: Brian
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:42:46 2020

@author: Brian
"""

#%% IMPORT
from IPython import get_ipython
get_ipython().magic('reset -sf')

import math as m
import numpy as np
import pandas as pd

# import os

import skimage
import skimage.measure
import skimage.draw
# from skimage.draw import set_color

from skimage.io import imread

from matplotlib.pyplot import imshow, show

# LOAD
RGB = imread('RGB.bmp')
RGBi = imread('RGBi.png')
Devices_Dfct = pd.read_pickle('Devices_Dfct')

#%% Randomly sample n rows from Devices_Dfct to create Devices_Training
Devices_Training = Devices_Dfct.sample(n=200, random_state=42, axis=0)

#%% add "label" column
Devices_Training = Devices_Training.reindex(columns = Devices_Training.columns.tolist() + ['label'])

#%% add a label for each defect type
Devices_Training = Devices_Training.reindex(columns = Devices_Training.columns.tolist() + ['p','nf','ed','eed','ene','enf','s'])
 
#%% change the nan's to 0's
Devices_Training.loc[:,['label','p','nf','ed','eed','ene','enf','s']] = 0 
#%% SAVE
Devices_Training.to_pickle('Devices_Training')