# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:22:36 2020

This code makes the Devices_Info DataFrame which
contains information about each device on the wafer
including: the square row, the square column,
the bounding box, and the coordinate list.

@author: Brian
"""

#%% IMPORT

from IPython import get_ipython
get_ipython().magic('reset -sf')

import math as m
import numpy as np
import pandas as pd

import skimage
import skimage.measure
import skimage.draw

from skimage.io import imread

#%% LOAD

# RGB = imread('RGB.bmp')
IND = imread('Images\IND.png')
mask_sqrs = imread('Images\mask_sqrs.png')
dam = imread('Images\dam.png')
# b_dam  = dam==255

#%% Figure out coordinates of centroids and fill up device coordiantes accordingly

LBL_sqrs = skimage.measure.label(mask_sqrs)
props_sqrs = skimage.measure.regionprops_table(LBL_sqrs,properties=['bbox','coords'])
props_sqrs = pd.DataFrame(props_sqrs)

#%% Create Devices_Info_Info Dataframe 

# Intialize dataframe
col_names = ['sqr_row','sqr_col','sqr_bbox','coords']
Devices_Info = pd.DataFrame(data=None,columns=col_names)

for i, row in props_sqrs.iterrows():
    this_bbox = row['bbox-0'],row['bbox-1'],row['bbox-2'],row['bbox-3']
    TLcorner = row['bbox-0'],row['bbox-1']
   
    coords = row['coords']
    
    I = m.floor(TLcorner[0]/(np.shape(mask_sqrs)[1]-100)*np.shape(dam)[1])
    J = m.floor(TLcorner[1]/(np.shape(mask_sqrs)[0]-100)*np.shape(dam)[0])
    Devices_Info = Devices_Info.append({'sqr_row':I,'sqr_col':J,'sqr_bbox':this_bbox,'coords':coords},ignore_index = True)

# Sort them according to row/col location   
Devices_Info = Devices_Info.sort_values(by=['sqr_row','sqr_col'],ascending=True)
#%% SAVE Devices Dataframe

Devices_Info.to_pickle('Data\Devices_Info')