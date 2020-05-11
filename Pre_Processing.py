# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:14:51 2020

This code is responsible for performing pre-processing 
procedures on the raw RGB dataset. The code uses 
"mask_sqrs.png" to mask out the device regions on the
wafer. The code then produces various image masks
to define various important colors on the wafer. From
these masks, and indexed image (IND) is formed which
is usutilized in the subsequent image processing. An
RGB representation of the indexed image (RGBi) is also
produced for visualization of the indexed image.

@author: Brian
"""

#%% import

import numpy as np

from skimage.io import imread
from skimage.io import imsave

from skimage.color import rgb2hsv

#%% LOAD

RGB = imread('Raw_Data\RGB.bmp')
mask_sqrs = imread('Images\mask_sqrs.png')
bool_sqrs = mask_sqrs/255
bool_sqrs = mask_sqrs.astype(np.bool)

#%% Save RGB as a jpg (to reduce filesize and use as visual aid later)

imsave('Images\RGB.jpg',RGB)

#%% mask to make RGBm

RGBm = np.zeros([np.shape(RGB)[0],np.shape(RGB)[1],np.shape(RGB)[2]],dtype=np.uint8)
RGBm[mask_sqrs==255] = RGB[mask_sqrs==255];

#%% convert RGBm to HSV and split out layers H, S, and V

# convert to HSV
HSVm = rgb2hsv(RGBm)*255
HSVm = HSVm.astype(np.uint8)

# split into layers
H = HSVm[:,:,0]
S = HSVm[:,:,1]
V = HSVm[:,:,2]

#%% create mask_yield (mask for yielded pixels)

# yield conditions
temp1 = H>60
temp2 = H<115
temp3 = S>140
temp4 = V>140

# make mask_yield
mask_yield = temp1 & temp2 & temp3 & temp4 & bool_sqrs
mask_yield = mask_yield.astype(np.uint8)*255

#%% create mask_k (mask for black colored pixels)

# black condition
temp1 = V<75

# make mask_k
mask_k = temp1 & bool_sqrs
mask_k = mask_k.astype(np.uint8)*255

#%% create mask_si (mask for si colored pixels)

# si condition
temp1 = V>75
temp2 = S<50

# make mask_si
mask_si = temp1 & temp2 & bool_sqrs
mask_si = mask_si.astype(np.uint8)*255

#%% make a mask for the remaining pixels...

bool_yield  = mask_yield.astype(np.bool)
bool_k  = mask_k.astype(np.bool)
bool_si = mask_si.astype(np.bool)

bool_remain = bool_sqrs & ~bool_yield & ~bool_k & ~bool_si 
mask_remain = bool_remain.astype(np.uint8)*255

#%% make mask_rd and mask_fg 

# convert to float
RGBf = RGB.astype(np.float64)

# red and faded green color centroids
rd = [120, 119, 55]
fg = [ 50,  90, 50]

# calculate root square error (RSE) of pixels with respect to each color centroid
RSE_rd = ( (RGBf[:,:,0]-rd[0])**2 + (RGBf[:,:,1]-rd[1])**2 + (RGBf[:,:,2]-rd[2])**2 )**0.5
RSE_fg = ( (RGBf[:,:,0]-fg[0])**2 + (RGBf[:,:,1]-fg[1])**2 + (RGBf[:,:,2]-fg[2])**2 )**0.5

# stack RSE_rd and RSE_fg into RSE 
RSE = np.zeros([np.shape(RGB)[0],np.shape(RGB)[1],2],dtype=np.float64)
RSE[:,:,0] = RSE_rd
RSE[:,:,1] = RSE_fg

# calculate minimum in RSE to determine which color centroid pixels were closet to
RSEmin = np.argmin(RSE,axis=2)
temp1 = RSEmin==0
temp2 = RSEmin==1

# make masks for red and faded green pixels
bool_rd = temp1 & bool_remain
mask_rd = bool_rd.astype(np.uint8)*255
bool_fg = temp2 & bool_remain
mask_fg = bool_fg.astype(np.uint8)*255

#%% create indexed image

# initialize array for indexed image
IND = np.zeros([np.shape(RGB)[0],np.shape(RGB)[1]],dtype=np.uint8)

# color index definitions
IND[mask_sqrs ==  0] = 0 # device index
IND[mask_yield==255] = 1 # yield index
IND[mask_k    ==255] = 2 # black index
IND[mask_si   ==255] = 3 # si index
IND[mask_rd   ==255] = 4 # red index
IND[mask_fg   ==255] = 5 # faded green index

# save indexed imaged (IND)
imsave('Images\IND.png',IND)

#%% create index RGB image (RGBi) (an RGB image with just the 6 colors defined by the indexing)

# initialize array for indexed RGB image
RGBi = np.zeros([np.shape(RGB)[0],np.shape(RGB)[1],np.shape(RGB)[2]],dtype=np.uint8)

# background
ind_bkg = np.where(mask_sqrs == 0)
RGBi[ind_bkg[0],ind_bkg[1],:] = [70,70,70]

# yield
ind_y = np.where(mask_yield == 255)
RGBi[ind_y[0],ind_y[1],:] = [0,255,0]

# black
ind_k = np.where(mask_k == 255)
RGBi[ind_k[0],ind_k[1],:] = [10,10,10]

# silicon
ind_si = np.where(mask_si == 255)
RGBi[ind_si[0],ind_si[1],:] = [150,150,200]

# red
ind_rd = np.where(mask_rd == 255)
RGBi[ind_rd[0],ind_rd[1],:] = [255,0,0]

# faded green
ind_fg = np.where(mask_fg == 255)
RGBi[ind_fg[0],ind_fg[1],:] = [255,255,0]

# save indexed RGB image
imsave('Images\RGBi.png',RGBi)