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

import skimage
import skimage.measure
import skimage.draw
# from skimage.draw import set_color

from skimage.io import imread

from matplotlib.pyplot import imshow, show

# LOAD
RGB = imread('RGB.bmp')
RGBi = imread('RGBi.png')
Devices_Training = pd.read_pickle('Devices_Training')

#%% ITERROWS Label

from random import seed
seed(1)
from random import choice

extra= 200
dims = np.shape(RGB)

count = 0
for index, row in Devices_Training.iterrows():
    count = count + 1
    # if np.isnan(Devices_Training.loc[index,"label_actual"]):
    if isinstance(Devices_Training.loc[index,"label"],str) == 0:
        
        #start labelling process otherwise keep looping
        print(count,index)
        this_bbox = Devices_Training.loc[index,'sqr_bbox']
        
        row_min = max(this_bbox[0]-extra, 0)
        row_max = min(this_bbox[2]+extra, dims[0])
        col_min = max(this_bbox[1]-extra, 0)
        col_max = min(this_bbox[3]+extra, dims[1])
    
        RGBir = RGBi.copy()
        
        # produce image which shows device in question
        start = (this_bbox[0], this_bbox[1])
        end   = (this_bbox[2], this_bbox[3])
        rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
        RGBir[rr, cc] = [255,0,0]
        start = (this_bbox[0]+1, this_bbox[1]+1)
        end   = (this_bbox[2]+1, this_bbox[3]+1)
        rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
        RGBir[rr, cc] = [0,255,255]        
        I_device = RGBir[row_min:row_max, col_min:col_max]        
        imshow(I_device)
        show()

        print('Enter the class...',
              'p (particle);',
              'nf (non-fill);',
              'ed (etch delay);',
              'eed (edge etch delay);',
              'ene (edge non-etch);',
              'enf (edge non-fill);',
              's (scratch);',
              'm (mixed);',
              'quit (to quit)')
        
        # inp = input()
        inp = str(input("Enter comma separated strs: "))
        if inp=='quit':
            break
        else:        
            Devices_Training.loc[index,'label_actual']  = inp
    #
#    
series = Devices_Training["label_actual"]
print(series.value_counts())

Devices_Training.to_pickle('Devices_Training') 

    
#%% go through and one-hot encode the labels                                                                             
for index, row in Devices_Training.iterrows():
    label = Devices_Training.loc[index,'label_actual']
    label_list = label.split (",")
    for i in range(np.shape(label_list)[0]):
        if label_list[i] == 'p':
            Devices_Training.loc[index,'p'  ] = 1
        if label_list[i] == 'nf':
            Devices_Training.loc[index,'nf' ] = 1
        if label_list[i] == 'ed':
            Devices_Training.loc[index,'ed' ] = 1
        if label_list[i] == 'eed':
            Devices_Training.loc[index,'eed'] = 1
        if label_list[i] == 'ene':
            Devices_Training.loc[index,'ene'] = 1
        if label_list[i] == 'enf':
            Devices_Training.loc[index,'enf'] = 1
        if label_list[i] == 's':
            Devices_Training.loc[index,'s'  ] = 1

#%%
Devices_Training.loc[:,'label'] = Devices_Training.loc[:,'label_actual']
Devices_Training = Devices_Training.drop(['label_actual'],axis=1)

#%% SAVE
# Devices_Training = Devices_Training
Devices_Training.to_pickle('Devices_Training')
#%% INVESTIGATE A DEVICE
RGB_real = imread('RGB.bmp')
index = 441
row_min = max(this_bbox[0]-extra, 0)
row_max = min(this_bbox[2]+extra, dims[0])
col_min = max(this_bbox[1]-extra, 0)
col_max = min(this_bbox[3]+extra, dims[1])

RGBr = RGB_real.copy()

# produce image which shows device in question
start = (this_bbox[0], this_bbox[1])
end   = (this_bbox[2], this_bbox[3])
rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
RGBr[rr, cc] = [255,0,0]
start = (this_bbox[0]+1, this_bbox[1]+1)
end   = (this_bbox[2]+1, this_bbox[3]+1)
rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
RGBr[rr, cc] = [255,0,0]        
I_device = RGBr[row_min:row_max, col_min:col_max]        
imshow(I_device)
show()

#%% Select specific device and label

extra= 200
dims = np.shape(RGB)

N_devices = np.shape(Devices_Training)[0]


index = 449

#start labelling process otherwise keep looping
print(index)
this_bbox = Devices_Training.loc[index,'sqr_bbox']

row_min = max(this_bbox[0]-extra, 0)
row_max = min(this_bbox[2]+extra, dims[0])
col_min = max(this_bbox[1]-extra, 0)
col_max = min(this_bbox[3]+extra, dims[1])

RGBr = RGB.copy()

# produce image which shows device in question
start = (this_bbox[0], this_bbox[1])
end   = (this_bbox[2], this_bbox[3])
rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
RGBr[rr, cc] = [255,0,0]
start = (this_bbox[0]+1, this_bbox[1]+1)
end   = (this_bbox[2]+1, this_bbox[3]+1)
rr, cc = skimage.draw.rectangle_perimeter(start, end=end)
RGBr[rr, cc] = [255,0,0]        
I_device = RGBr[row_min:row_max, col_min:col_max]        
imshow(I_device)
show()

print('Enter the class...',
      'p (particle);',
      'nf (non-fill);',
      'ed (etch delay);',
      'eed (edge etch delay);',
      'ene (edge non-etch);',
      'enf (edge non-fill);',
      's (scratch);',
      'm (mixed);',
      'quit (to quit)')

inp = input()           
Devices_Training.loc[index,'label']  = inp
#
#    
# Devices_Training.insert(np.shape(Devices_Dfct)[1],'label',labels)

# Devices_Training.to_pickle('Devices_Training') 
