# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:24:56 2020

This code creates the Devices_Dfct dataframe which
contains information and features about all the defective
devices (devices with >10% defectivity).

It starts by cacluating the overall fraction of each
device that is defective, and then puts devices that are
>10% defective into a dataframe which will be used for
subsequent analysis (we don't care about devices with 
<10% defectivity). Then, the code proceeds to calculate a
number of other features for the devices and adds these
to the dataframe.

@author: Brian
"""
#%% IMPORT

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

from skimage.io import imread

from skimage.morphology import binary_dilation

#%% LOAD

# RGB  = imread('Images\RGB.bmp')
Devices_Info = pd.read_pickle('Data\Devices_Info')
IND = imread('Images\IND.png')
mask_sqrs  = imread('Images\mask_sqrs.png')
dam = imread('Images\dam.png')
dem = imread('Images\dem.png')

#%% Make fraction defective feature

print('Making frac_dfct feature...')

Devices_Info = Devices_Info.reindex(columns = Devices_Info.columns.tolist() + ['frac_dfct'])

dims = np.shape(mask_sqrs)

for index, row in Devices_Info.iterrows():
    
    this_bbox = row['sqr_bbox']
    row_min = max(this_bbox[0], 0      )
    row_max = min(this_bbox[2], dims[0])
    col_min = max(this_bbox[1], 0      )
    col_max = min(this_bbox[3], dims[1])
    d_IND = IND[row_min:row_max,col_min:col_max]
    # d_RGB = RGB[row_min:row_max,col_min:col_max]
    
    N_yield = np.sum(d_IND==1) #yield
    
    N_tot = np.size(d_IND) # total number of pixels in square
    N_dfct = N_tot - N_yield # number of defective pixels in square
    
    Devices_Info.loc[index,'frac_dfct'] = N_dfct/N_tot

#%% Make Defective Devices Table

# Defective devices are taken as devices with > 10% of their pixels defective
Devices_Dfct = Devices_Info.loc[Devices_Info['frac_dfct'] >= 0.10]

#%% Make fraction color features

print('Making fraction color features...')

Devices_Dfct = Devices_Dfct.reindex(
                 columns = Devices_Dfct.columns.tolist() + ['frac_k' ,
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
    # d_RGB = RGB[row_min:row_max,col_min:col_max]
    
    N_yield = np.sum(d_IND==1) #yield
    
    N_tot = np.size(d_IND) # total number of pixels in square
    N_dfct = N_tot - N_yield # number of defective pixels in square
    
    Devices_Dfct.loc[index,'frac_k' ] = np.sum(d_IND==2)/N_dfct # black
    Devices_Dfct.loc[index,'frac_si'] = np.sum(d_IND==3)/N_dfct # si
    Devices_Dfct.loc[index,'frac_rd'] = np.sum(d_IND==4)/N_dfct # red
    Devices_Dfct.loc[index,'frac_fg'] = np.sum(d_IND==5)/N_dfct # faded green
#%%   Make edge_yn feature
    
print('Making edge_yn feature...')

Devices_Dfct = Devices_Dfct.reindex(columns = Devices_Dfct.columns.tolist() + ['edge_yn'])

for index, row in Devices_Dfct.iterrows():
    # print(index)
    r,c = row['sqr_row'],row['sqr_col']
    # c = row['sqr_col']
    if dem[r,c]==255:
        Devices_Dfct.loc[index,'edge_yn'] = 1
    else:
        Devices_Dfct.loc[index,'edge_yn'] = 0

#%%   Make perimeter fraction features
        
print('Making perimeter fraction features..')

'''calculates fraction of perimeter of si regions
overlapping with a specific color.
'''
def si_p_frac(index,IND,Devices_Dfct,color):
    
    dims = np.shape(IND)
    
    # get d_IND
    this_bbox = Devices_Dfct.loc[index,'sqr_bbox']
    row_min = max(this_bbox[0], 0      )
    row_max = min(this_bbox[2], dims[0])
    col_min = max(this_bbox[1], 0      )
    col_max = min(this_bbox[3], dims[1])
    d_IND = IND[row_min:row_max,col_min:col_max]
        
    # Get masks (change numbers)
    m_si  = d_IND==3
    m_c   = d_IND==color

    # Dilate the mask Si image
    m_si_dil = binary_dilation(m_si)
    
    # Make perimiter pixels image
    m_perim = m_si_dil & np.invert(m_si) # Mask containing just the outer perimeter pixels
    N_perim_tot = np.sum(m_perim) # Number of total outer perimeter pixels
    
    # Get mask image of perim for this color
    m_perim_c  = m_perim & m_c

    # Calculate number of outer perimeter pixels overlapping with this color
    N_perim_c  = np.sum(m_perim_c)

    # Calculate fraction outer perimeter pixels overlapping with this color
    p_frac_c  = N_perim_c / N_perim_tot
    
    # Check for NaN's and set them = 0
    test = np.isnan(p_frac_c)
    if test == 1:
        p_frac_c = 0
    
    return p_frac_c

Devices_Dfct = Devices_Dfct.reindex(
                columns = Devices_Dfct.columns.tolist() + ['p_frac_y',
                                                           'p_frac_k',
                                                           'p_frac_r',
                                                           'p_frac_fg'])

# Calculate p frac for various colors and add the results to Devices_Dfct
for index, row in Devices_Dfct.iterrows():

    p_frac_y  = si_p_frac(index,IND,Devices_Dfct,1)
    p_frac_k  = si_p_frac(index,IND,Devices_Dfct,2)
    p_frac_r  = si_p_frac(index,IND,Devices_Dfct,4)
    p_frac_fg = si_p_frac(index,IND,Devices_Dfct,5)
    
    Devices_Dfct.loc[index,'p_frac_y' ] = p_frac_y
    Devices_Dfct.loc[index,'p_frac_k' ] = p_frac_k
    Devices_Dfct.loc[index,'p_frac_r' ] = p_frac_r
    Devices_Dfct.loc[index,'p_frac_fg'] = p_frac_fg
        
#%%   SAVE Devices_Dfct_Info and Devices_Dfct_Features
    
print('Saving...')

Devices_Dfct_Info = Devices_Dfct.filter(items=['sqr_row', 'sqr_col','sqr_bbox','coords'])
Devices_Dfct_Features = Devices_Dfct.drop(['sqr_row', 'sqr_col','sqr_bbox','coords'],axis=1)

Devices_Dfct_Info.to_pickle('Data\Devices_Dfct_Info')
Devices_Dfct_Features.to_pickle('Data\Devices_Dfct_Features')
