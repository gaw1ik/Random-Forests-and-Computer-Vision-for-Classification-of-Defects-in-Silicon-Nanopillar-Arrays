# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:41:29 2020

@author: Brian
"""
#%% IMPORT
from IPython import get_ipython
get_ipython().magic('reset -sf')

# import math as m
import numpy as np
import pandas as pd

# import skimage

from skimage.io import imread
# from skimage.io import imshow

from skimage.morphology import binary_dilation
def perimeter_fraction(index,IND,Devices):
    dims = np.shape(IND)
    # index = 1
    
    # get d_IND
    this_bbox = Devices_Dfct.loc[index,'sqr_bbox']
    row_min = max(this_bbox[0], 0      )
    row_max = min(this_bbox[2], dims[0])
    col_min = max(this_bbox[1], 0      )
    col_max = min(this_bbox[3], dims[1])
    d_IND = IND[row_min:row_max,col_min:col_max]
    # d_RGB = RGB[row_min:row_max,col_min:col_max]
        
    # Get masks (change numbers)
    m_si  = d_IND==3
    m_y   = d_IND==1
    m_k   = d_IND==2
    m_fg  = d_IND==5

    # Get perims
    m_si_dil = binary_dilation(m_si) # Dilate the mask Si image
    m_si_dil = binary_dilation(m_si_dil) # Dilate the mask Si image
    
    # make perimiter pixels image
    m_perim = m_si_dil & np.invert(m_si) # Mask containing just the outer perimeter pixels
    N_perim_tot = np.sum(m_perim) # Number of total outer perimeter pixels
    
    # Get mask images of perims for different color values
    m_perim_y  = m_perim & m_y # for yield
    m_perim_k  = m_perim & m_k
    m_perim_fg = m_perim & m_fg
    
    # Calculate number of outer perimeter pixels overlapping with various colors
    N_perim_y  = np.sum(m_perim_y) # for yield
    N_perim_k  = np.sum(m_perim_k)
    N_perim_fg = np.sum(m_perim_fg)

    # Calculate fraction outer perimeter pixels overlapping with various colors
    p_frac_y  = N_perim_y /N_perim_tot # for yield
    p_frac_k  = N_perim_k /N_perim_tot
    p_frac_fg = N_perim_fg/N_perim_tot
    
    # Check for NaN's and set them = 0
    test = np.isnan(p_frac_y)
    if test == 1:
        p_frac_y = 0
    
    test = np.isnan(p_frac_k)
    if test == 1:
        p_frac_k = 0
    
    test = np.isnan(p_frac_fg)
    if test == 1:
        p_frac_fg = 0
    
    return p_frac_y, p_frac_k, p_frac_fg

#%% LOAD
Devices = pd.read_pickle('Devices')
IND  = imread( 'IND.png')
RGB  = imread( 'RGB.bmp')
RGBi = imread('RGBi.png')
mask_sqrs  = imread( 'mask_sqrs.png')
dam  = imread('dam.png')
dem  = imread('dem.png')
b_dam  = dam==255

#%% Pull each device IND image, calculate features, and add to Devices
Devices = Devices.reindex(columns = Devices.columns.tolist() + ['frac_dfct'])

dims = np.shape(mask_sqrs)
print('frac_dfct')
for index, row in Devices.iterrows():
    
    
    this_bbox = row['sqr_bbox']
    row_min = max(this_bbox[0], 0      )
    row_max = min(this_bbox[2], dims[0])
    col_min = max(this_bbox[1], 0      )
    col_max = min(this_bbox[3], dims[1])
    d_IND = IND[row_min:row_max,col_min:col_max]
    d_RGB = RGB[row_min:row_max,col_min:col_max]
    
    N_yield = np.sum(d_IND==1) #yield
    
    N_tot = np.size(d_IND) # total number of pixels in square
    N_dfct = N_tot - N_yield # number of defective pixels in square
    
    Devices.loc[index,'frac_dfct'] = N_dfct/N_tot

#%%   Make Defective Devices Table
Devices_Dfct = Devices.loc[Devices['frac_dfct'] >= 0.10]
# Devices_Dfct.to_pickle('Devices_Dfct')
#%% make frac features
print('frac_k, frac_si, frac_rd, frac_fg')

Devices_Dfct = Devices_Dfct.reindex(columns = Devices_Dfct.columns.tolist() + ['frac_k',
                                                                'frac_si',
                                                                'frac_rd',
                                                                'frac_fg'])

dims = np.shape(mask_sqrs)

for index, row in Devices_Dfct.iterrows():
    # print(index)    
    this_bbox = row['sqr_bbox']
    row_min = max(this_bbox[0], 0      )
    row_max = min(this_bbox[2], dims[0])
    col_min = max(this_bbox[1], 0      )
    col_max = min(this_bbox[3], dims[1])
    d_IND = IND[row_min:row_max,col_min:col_max]
    d_RGB = RGB[row_min:row_max,col_min:col_max]
    
    N_yield = np.sum(d_IND==1) #yield
    
    N_tot = np.size(d_IND) # total number of pixels in square
    N_dfct = N_tot - N_yield # number of defective pixels in square
    
    Devices_Dfct.loc[index,'frac_k'] = np.sum(d_IND==2)/N_dfct #black
    Devices_Dfct.loc[index,'frac_si'] = np.sum(d_IND==3)/N_dfct #si
    Devices_Dfct.loc[index,'frac_rd'] = np.sum(d_IND==4)/N_dfct #red
    Devices_Dfct.loc[index,'frac_fg'] = np.sum(d_IND==5)/N_dfct #fg
#%%   Make edge_yn feature
print('edge_yn')

Devices_Dfct = Devices_Dfct.reindex(columns = Devices_Dfct.columns.tolist() + ['edge_yn'])

for index, row in Devices_Dfct.iterrows():
    # print(index)
    r,c = row['sqr_row'],row['sqr_col']
    # c = row['sqr_col']
    if dem[r,c]==255:
        Devices_Dfct.loc[index,'edge_yn'] = 1
    else:
        Devices_Dfct.loc[index,'edge_yn'] = 0

#%%   Make p_frac features
print('p_frac')
Devices_Dfct = Devices_Dfct.reindex(columns = Devices_Dfct.columns.tolist() + ['p_frac_y',
                                                                               'p_frac_k',
                                                                               'p_frac_fg'])

for index, row in Devices_Dfct.iterrows():
    # print(index)
    p_frac_y, p_frac_k, p_frac_fg = perimeter_fraction(index,IND,Devices_Dfct)
    Devices_Dfct.loc[index,'p_frac_y' ] = p_frac_y
    Devices_Dfct.loc[index,'p_frac_k' ] = p_frac_k
    Devices_Dfct.loc[index,'p_frac_fg'] = p_frac_fg
        
#%%   SAVE Devices_Dfct
Devices_Dfct.to_pickle('Devices_Dfct')
