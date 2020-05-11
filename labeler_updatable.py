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

#%% LOAD

RGB = imread('Images\RGB.jpg')
RGBi = imread('Images\RGBi.png')

#%% LOAD IN THE CORRECT DEVICES_TRAINING

Devices_Training_Info = pd.read_pickle('Data\Devices_Training_Info')
                                       
inp = str(input("Are you updating (enter 1) or starting fresh (enter 2)? INPUT: "))
if   inp=='1':
    Label_Lists = pd.read_pickle('Data\Label_Lists')
elif inp=='2':    
    # Make a Fresh Labels (this needs to be updated)
    Label_Lists = pd.DataFrame(columns=['label_list'])
    # add "label" column
    # Labels = Labels.reindex(columns = Devices_Training_Unlabeled.columns.tolist() + ['label'])
    # add a label for each defect type
    # Labels = Labels.reindex(columns = Devices_Training_Unlabeled.columns.tolist() + ['p','nf','ed','eed','ene','enf','s'])
    # change the nan's to 0's
    # Labels.loc[:,['label','p','nf','ed','eed','ene','enf','s']] = 0 
    # Devices_Training = pd.read_pickle('Devices_Training_Unlabeled')
else:
    print('unexpected answer. failed to load a training dataset.')

#%% Rename stuff

# Label_Lists = Label_Lists.rename(columns={"label": "label_list"})

#%% ITERROWS Label

extra= 200
dims = np.shape(RGB)

count = 0
for index, row in Label_Lists.iterrows():
    count = count + 1
    # if np.isnan(Devices_Training.loc[index,"label_actual"]):
    if isinstance(Label_Lists.loc[index,"label_list"],str) == 0:
        
        #start labelling process otherwise keep looping
        print(count,index)
        this_bbox = Devices_Training_Info.loc[index,'sqr_bbox']
        
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
            Label_Lists.loc[index,'label_list']  = inp
    #
#    
# series = Label_Lists["label_list"]
# print(series.value_counts())

#%% SAVE (think twice before you save so you don't overwrite)

Label_Lists.to_pickle('Data\Label_Lists') 
    
#%% Go through and one-hot encode the labels and make Devices_Training_Labels

zers = np.zeros([200,7])

Devices_Training_Labels = pd.DataFrame(data=zers,index = Label_Lists.index,columns = ['p','nf','ed','eed','ene','enf','s'])
# Devices_Training_Labels.loc[:,['p','nf','ed','eed','ene','enf','s']] = 0 
                                                                  
for index, row in Label_Lists.iterrows():
    label = row['label_list']
    label_list = label.split (",")
    for label in label_list:
        if label == 'p':
            Devices_Training_Labels.loc[index,'p'  ] = 1
        elif label == 'nf':
            Devices_Training_Labels.loc[index,'nf' ] = 1
        elif label == 'ed':
            Devices_Training_Labels.loc[index,'ed' ] = 1
        elif label == 'eed':
            Devices_Training_Labels.loc[index,'eed'] = 1
        elif label == 'ene':
            Devices_Training_Labels.loc[index,'ene'] = 1
        elif label == 'enf':
            Devices_Training_Labels.loc[index,'enf'] = 1
        elif label == 's':
            Devices_Training_Labels.loc[index,'s'  ] = 1
        else:
            print(index,'warning: unrecognized label')


#%% SAVE

Devices_Training_Labels.to_pickle('Data\Devices_Training_Labels')