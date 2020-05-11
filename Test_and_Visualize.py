# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:31:27 2020

This code will use the RF model to predict 
classifications for ALL of the defective devices. 
Then it will output some sort of visualization to show what
classifications it has made.

@author: Brian
"""

#%% IMPORT

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

from skimage.io import imread, imsave

import pickle

#%% LOAD

# Load RGB
RGB = imread('Images\RGB.jpg')

# Load in full Devices_Dfct_ Data Frames
Devices_Dfct_Info = pd.read_pickle('Data\Devices_Dfct_Info')
Devices_Dfct_Features = pd.read_pickle('Data\Devices_Dfct_Features')

# Load the classifier model
clf = pickle.load(open('clf.sav', 'rb'))

#%% Drop certain features

# Devices_Dfct_Features = Devices_Dfct_Features.drop(['p_frac_r'],axis=1)

#%% Use the classifier model to make predictions

X = Devices_Dfct_Features

y_predict = clf.predict(X)
y_predict = pd.DataFrame(data=y_predict,index=X.index,columns=['p_predict','nf_predict','ed_predict','eed_predict','ene_predict','enf_predict','s_predict'])

# Add the prediction labels to the dataframe
Devices_Test = pd.concat([Devices_Dfct_Info, y_predict], axis=1)

# SAVE Devices_Test
Devices_Test.to_pickle('Data\Devices_Test')

#%% MAKE GRAY IMAGE

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

gray = make_gray_rgb(RGB)

#%% MAKE VISUALIZATIONS

''' This function makes a classification image. '''
def make_class_image(Devices_Dfct,RGB,gray,defect_class):
    
    Devices_dc = Devices_Dfct.loc[Devices_Dfct[defect_class] == 1]
    gray_dc = gray.copy()
    
    for index, row in Devices_dc.iterrows():
        coords = row['coords']
        x = coords[:,0]
        y = coords[:,1]
        gray_dc[x,y] = RGB[x,y] 
        
    return gray_dc

# Produce and save the classifaction 
# images for each defect type.
dfct_types = ['p_predict'  ,'nf_predict' ,
              'ed_predict' ,'eed_predict',
              'ene_predict','enf_predict','s_predict']

# dfct_types = ['p_predict'] # for testing

classifaction_ims = []

for dfct_type in dfct_types:
    print('making classification image for',dfct_type)
    im = make_class_image(Devices_Test,RGB,gray,dfct_type)
    classifaction_ims.append(im)
    imsave('Figures\classification_image_' + dfct_type + '.jpg',im)