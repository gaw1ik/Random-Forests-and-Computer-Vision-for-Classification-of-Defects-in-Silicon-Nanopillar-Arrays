# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:45:27 2020

@author: Brian
"""

#%% IMPORT
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os 
import numpy as np
import pandas as pd

from skimage.io import imread
# from skimage.io import imshow
from matplotlib.pyplot import imshow, show

# LOAD
Devices_Training = pd.read_pickle('Devices_Training')
# Devices_Dfct = pd.read_pickle('Devices_Dfct')

#%% take only the rows that have been labeled
# Devices_Training = Devices_Training.iloc[0:29,:]
Devices_Training = Devices_Training.dropna()

#%% Create X and y, SPLIT into train/eval, encode
from sklearn.model_selection import train_test_split

# Devices_Training_Enc = Devices_Training

#encode
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(["p", "nf", "ed", "eed","ene","enf","s","m"])
# le.classes_
# Devices_Training_Enc['label_actual'] = le.transform(Devices_Training['label_actual']) 

X = Devices_Training.drop(['label','p','nf','ed','eed','ene','enf','s'],axis=1)

y = Devices_Training[['p','nf','ed','eed','ene','enf','s']]

X_train_w_info, X_eval_w_info, y_train, y_eval = train_test_split(
     X, y, test_size=0.33, random_state=42)

#%% Train and Predict
# Drop info columns 
X_train = X_train_w_info.drop(['sqr_row',
                               'sqr_col',
                               'sqr_bbox',
                               'coords'],axis=1)

X_eval = X_eval_w_info.drop(['sqr_row',
                             'sqr_col',
                             'sqr_bbox',
                             'coords'],axis=1)

from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state = 42)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state = 42)

# Train the model on training data
clf.fit(X_train, y_train)

#%% EVALUATE and SCORE
# Devices_Eval = X_eval_w_info.copy()

# Devices_Eval = pd.concat([X_eval_w_info, y_eval, y_predict], axis=1)

#%%

# Devices_Eval = Devices_Eval.reindex(columns = Devices_Eval.columns.tolist() + ['label','label_prediction','correct?'])

y_predict = clf.predict(X_eval)
y_predict = pd.DataFrame(data=y_predict,index=y_eval.index,columns=['p_predict','nf_predict','ed_predict','eed_predict','ene_predict','enf_predict','s_predict'])
# Create Devices_Eval which includes my labels and prediction labels
Devices_Eval = pd.concat([X_eval_w_info, y_eval, y_predict], axis=1)

#SCORE
N_tot = np.size(y_eval) #number of entries in y_eval
y_eval_np = y_eval.to_numpy()
y_predict_np = y_predict.to_numpy()

def analysis1(y_eval_np,y_predict_np,bool1,bool2):
    b1 = y_eval_np==bool1
    b2 = y_predict_np==bool2
    N1 = np.sum(b1) #number of (positives) in y_eval
    b3 = b1 & b2
    N3 = np.sum(b3) #number of (true positives)
    perc = round( N3/N1*100 )
    return N3, perc
    
# True Positives
N3, perc = analysis1(y_eval_np,y_predict_np,1,1)
print('True Positives: ' + str(perc) +'%')
tp = N3
# True Negatives
N3, perc = analysis1(y_eval_np,y_predict_np,0,0)
print('True Negatives: ' + str(perc) +'%')
# Fals Positives
N3, perc = analysis1(y_eval_np,y_predict_np,1,0)
print('Fals Positives: ' + str(perc) +'%')
fp = N3
# Fals Negatives
N3, perc = analysis1(y_eval_np,y_predict_np,0,1)
print('Fals Negatives: ' + str(perc) +'%')
fn = N3

def precision(tp,fp):
	pr = round( tp/(tp+fp) * 100 )
	return pr
def recall(tp,fn):
	rc = round( tp/(tp+fn) * 100 )
	return rc

pr = precision(tp,fp)
print('Precision: ' + str(pr) +'%')
rc = recall   (tp,fn)
print('Recall   : ' + str(rc) +'%')
#%% Score individual defect types
def analysis(str1,str2,bool1,bool2):

	p_actual    = Devices_Eval.loc[:,str1]
	p_predicted = Devices_Eval.loc[:,str2] 

	b1 = p_predicted==bool1
	b2 = p_actual   ==bool2
	
	b3 = b1 & b2 #array of (true_positives)

	N1 = np.sum(b1) #number of predicted (positives)
	N3 = np.sum(b3) #number of (true_positives)

	perc = N3/N1 #percent of (true positives)

	return N3, perc

index_list = ['p','nf','ed','eed','ene','enf','s']

scores = pd.DataFrame(columns=['perc_true_pos','perc_true_neg','perc_fals_pos','perc_fals_neg'],
index=index_list)

scores2 = pd.DataFrame(columns=['precision','recall'],index=index_list)


for index, row in scores.iterrows():
	index_predict = index + '_predict'
	tp, scores.loc[index,'perc_true_pos'] = analysis(index,index_predict,1,1)
	fp, scores.loc[index,'perc_true_neg'] = analysis(index,index_predict,0,0)
	fp, scores.loc[index,'perc_fals_pos'] = analysis(index,index_predict,1,0)
	fn, scores.loc[index,'perc_fals_neg'] = analysis(index,index_predict,0,1)
	scores2.loc[index,'precision'] = precision(tp,fp)
	scores2.loc[index,'recall'   ] = recall   (tp,fn)

print(scores2.head(7))

