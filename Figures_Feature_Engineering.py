# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:29:09 2020

@author: Brian
"""

#%% IMPORT
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os 
import numpy as np
import pandas as pd

from skimage.io import imread, imshow, imsave

from skimage.transform import resize
import skimage.draw
from matplotlib.pyplot import show
from skimage.morphology import binary_dilation


import pickle

#%% LOAD
# LOAD in full Devices_Dfct
Devices_Dfct_Test = pd.read_pickle('Devices_Dfct_Test')
RGB = imread('RGB.bmp')
RGBi = imread('RGBi.png')
IND  = imread( 'IND.png')

#%% Make Training Set Image

# LOAD

RGB = imread('RGB.bmp')
Devices_Training_Info = pd.read_pickle('Data\Devices_Training_Info')


# MAKE GRAY IMAGE

''' make gray image from RGB 
(sets pixel values to local luminance) '''
def make_gray_rgb(RGB):
    R = RGB[:,:,0]
    G = RGB[:,:,1]
    B = RGB[:,:,2]
    L = np.uint8((0.2126*R + 0.7152*G + 0.0722*B)*0.25) # standard luminance calculation
    # gr = np.mean(RGB, axis=2, dtype = np.uint8) #avg level of 3 colors
    gray =  np.stack([L,L,L],axis=2)
    
    return gray
gray_train = make_gray_rgb(RGB)

for index, row in Devices_Training_Info.iterrows():
    coords = row['coords']
    x = coords[:,0]
    y = coords[:,1]
    gray_train[x,y] = RGB[x,y] 
    
imshow(gray_train)
imsave('Figures\gray_train.jpg',gray_train)

#%% look at device by [row,col] address
# inputs
# row,col = 19,13 # particle
# row,col = 31,5 # particle
row,col = 31,5 # particle
RGBir = RGBi.copy() # this sets the type of image that gets used (RGB, RGBi,etc)
extra = 0 # border space around the device

# find the row of Devices_Dfct_Test corresponding to that device and get its index
test = Devices_Dfct_Test.loc[(Devices_Dfct_Test['sqr_row'] == row) & (Devices_Dfct_Test['sqr_col'] == col)]
index = test.index[0]

dims = np.shape(RGB)

this_bbox = test.loc[index,'sqr_bbox']

row_min = max( (this_bbox[0] - extra), 0       )
row_max = min( (this_bbox[2] + extra), dims[0] )
col_min = max( (this_bbox[1] - extra), 0       )
col_max = min( (this_bbox[3] + extra), dims[1] )

# produce image which shows device in question     
d_RGB = RGBir[row_min:row_max, col_min:col_max]        
imshow(d_RGB)

#%% look at all particle detected devices from test set
# inputs
RGBir = RGBi.copy() # this sets the type of image that gets used (RGB, RGBi,etc)
extra = 0 # border space around the device

# find the rows of Devices_Dfct_Test with particles predicted
test = Devices_Dfct_Test.loc[(Devices_Dfct_Test['p_predict'] == 1)]

index = test.index[0]

dims = np.shape(RGB)

for index, row in test.iterrows():
    this_bbox = row['sqr_bbox']
  
    row_min = max( (this_bbox[0] - extra), 0       )
    row_max = min( (this_bbox[2] + extra), dims[0] )
    col_min = max( (this_bbox[1] - extra), 0       )
    col_max = min( (this_bbox[3] + extra), dims[1] )
    
    # produce image which shows device in question     
    d_RGB = RGBir[row_min:row_max, col_min:col_max]        
    imshow(d_RGB)
    show()
    
    inp = input('index = ' + str(index) + ', press enter to move to next image...')
    if inp=='quit':
        break

# devices i liked:
# 245,277,298,363,405(nf),500,556(p total knockout),574,607
#%%

# Devices = Devices.reindex(columns = Devices.columns.tolist() + ['frac_dfct'])
# index = 700
# index = 1093

dims = np.shape(IND)
   
# get d_IND
this_bbox = Devices_Dfct_Test.loc[index,'sqr_bbox']
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

test = np.isnan(p_frac_y)
if test == 1:
    p_frac_y = 0
    
test = np.isnan(p_frac_k)
if test == 1:
    p_frac_k = 0
    
test = np.isnan(p_frac_fg)
if test == 1:
    p_frac_fg = 0

imshow(m_perim_fg)
#%% testing
# imshow(m_si)

rr,cc = np.where(m_perim==1)
d_perim_RGB = np.zeros([103, 103, 3],dtype=np.uint8)
d_perim_RGB = d_perim_RGB + 255
#%%
d_perim_RGB[rr,cc] = d_RGB[rr,cc]
imshow(d_perim_RGB)
#%%

color = 5
    
dims = np.shape(IND)
# index = 1

# get d_IND
this_bbox = Devices_Dfct_Test.loc[index,'sqr_bbox']
row_min = max(this_bbox[0], 0      )
row_max = min(this_bbox[2], dims[0])
col_min = max(this_bbox[1], 0      )
col_max = min(this_bbox[3], dims[1])
d_IND = IND[row_min:row_max,col_min:col_max]
# d_RGB = RGB[row_min:row_max,col_min:col_max]
    
# Get masks (change numbers)
m_si  = d_IND==3
m_c   = d_IND==color

# Dilate
m_si_dil = binary_dilation(m_si) # Dilate the mask Si image
m_si_dil = binary_dilation(m_si_dil) # Dilate the mask Si image

# make perimiter pixels image
m_perim = m_si_dil & np.invert(m_si) # Mask containing just the outer perimeter pixels
N_perim_tot = np.sum(m_perim) # Number of total outer perimeter pixels

# Get mask image of perim for this color
m_perim_c  = m_perim & m_c

# Calculate number of outer perimeter pixels overlapping with this color
N_perim_c  = np.sum(m_perim_c) # for yield

# Calculate fraction outer perimeter pixels overlapping with various colors
p_frac_c  = N_perim_c / N_perim_tot # for yield

# Check for NaN's and set them = 0
test = np.isnan(p_frac_c)
if test == 1:
    p_frac_c = 0

imshow(m_perim_c)
print(p_frac_c)



