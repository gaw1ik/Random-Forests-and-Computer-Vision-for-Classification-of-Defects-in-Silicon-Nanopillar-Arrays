# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:25:56 2020

This code trains a random forest classifier and then the
classifier is evaluated against an evaluation dataset.
Various metrics quantifying
the accuracy etc. of the evalutation are calculated 
and printed out.

@author: Brian
"""

#%% IMPORT

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

import pickle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

#%% LOAD

Devices_Training_Info2 = pd.read_pickle('Data\Devices_Training_Info')
Devices_Training_Features2 = pd.read_pickle('Data\Devices_Training_Features')
Devices_Training_Labels2 = pd.read_pickle('Data\Devices_Training_Labels')

#%% For Pulling in the Old Devices_Training and just saving the labels as Devices_Training_Labels
# Devices_Training = pd.read_pickle('Devices_Training')
# Devices_Training_Labels = Devices_Training.filter(items=['label','p','nf','ed','eed','ene','enf','s'])
# Devices_Training_Labels.to_pickle('Data\Devices_Training_Labels')

#%% Take only the rows that have been labeled

# Devices_Training_Labels2 = Devices_Training_Labels.dropna()
# Devices_Training_Labels2 = Devices_Training_Labels2.drop(['label'],axis=1)

# Take just the rows from the DFs that were labeled training examples
# indexes = Devices_Training_Labels2.index
# Devices_Training_Features2 = Devices_Training_Features.loc[indexes,:]
# Devices_Training_Info2 = Devices_Training_Info.loc[indexes,:]

#%% Drop certain features

# Devices_Training_Features2 = Devices_Training_Features2.drop(['p_frac_r'],axis=1)

#%% Create X and y, SPLIT into train/eval

X = Devices_Training_Features2
y = Devices_Training_Labels2

X_train, X_eval, y_train, y_eval = train_test_split(
      X, y, test_size=0.33, random_state=42)

#%% Train

clf = RandomForestClassifier(random_state = 42)

# Train the model on training data
clf.fit(X_train, y_train)

# Save the classifier model
pickle.dump(clf, open('clf.sav', 'wb'))

#%% EVALUATE AND SCORE OVERALL (for all types of defects)

y_predict_np = clf.predict(X_eval)
y_predict = pd.DataFrame(data=y_predict_np,index=y_eval.index,columns=['p_predict','nf_predict','ed_predict','eed_predict','ene_predict','enf_predict','s_predict'])

# SCORE
N_tot = np.size(y_eval) #number of entries in y_eval
y_eval_np = y_eval.to_numpy()
# y_predict_np = y_predict.to_numpy()

''' calculates scores for evaulation 
pedictions for one of the following: 
-true positives  (1,1)
-true negatives  (0,0)
-false positives (1,0)
-false negatives (0,1)
'''
def overall_score(y_eval_np,y_predict_np,bool1,bool2):
    b1 = y_eval_np==bool1
    b2 = y_predict_np==bool2
    N1 = np.sum(b1) #number of (positives) in y_eval
    b3 = b1 & b2
    N3 = np.sum(b3) #number of (true positives)
    perc = round( N3/N1*100 )
    return N3, perc
    
# True Positives
N3, perc = overall_score(y_eval_np,y_predict_np,1,1)
# print('True Positives: ' + str(perc) +'%')
tp = N3

# True Negatives
N3, perc = overall_score(y_eval_np,y_predict_np,0,0)
# print('True Negatives: ' + str(perc) +'%')

# Fals Positives
N3, perc = overall_score(y_eval_np,y_predict_np,1,0)
# print('Fals Positives: ' + str(perc) +'%')

fp = N3
# Fals Negatives
N3, perc = overall_score(y_eval_np,y_predict_np,0,1)
# print('Fals Negatives: ' + str(perc) +'%')
fn = N3

''' calcualtes precision '''
def precision(tp,fp):
	pr = round( tp/(tp+fp) * 100 )
	return pr

''' calcualtes recall '''
def recall(tp,fn):
	rc = round( tp/(tp+fn) * 100 )
	return rc

# calculate overall precision and recall and print results
pr = precision(tp,fp)
rc = recall   (tp,fn)
# print('Precision: ' + str(pr) +'%')
# print('Recall   : ' + str(rc) +'%')

#%% EVAULATE AND SCORE FOR INDIVIDUAL DEFECT TYPES

# Create Devices_Eval which includes my labels and prediction labels
Devices_Eval = pd.concat([y_eval, y_predict], axis=1)

''' calculates scores for evaulation 
pedictions for a specific defect type 
for one of the following: 
-true positives  (1,1)
-true negatives  (0,0)
-false positives (1,0)
-false negatives (0,1)
'''
def score(defect_type,defect_type_predict,bool1,bool2):

	p_actual    = Devices_Eval.loc[:,defect_type]
	p_predicted = Devices_Eval.loc[:,defect_type_predict] 

	b1 = p_predicted==bool1
	b2 = p_actual   ==bool2
	
	b3 = b1 & b2 # array of (true_positives, for instance)

	N1 = np.sum(b1) # number of predicted (positives, for instance)
	N3 = np.sum(b3) # number of (true_positives, for instance)

	perc = N3/N1 # percent of (true positives, for instance)

	return N3, perc

defect_types = ['p','nf','ed','eed','ene','enf','s']

scores = pd.DataFrame(columns=['perc_true_pos','perc_true_neg','perc_fals_pos','perc_fals_neg'],
index=defect_types)

scores2 = pd.DataFrame(columns=['# of training examples','precision','recall'],index=defect_types)

defect_types = ['p','nf','ed','eed','ene','enf','s']
for defect_type in defect_types:
    test = Devices_Training_Labels2[defect_type].tolist()
    scores2.loc[defect_type,'# of training examples'] = sum(test)
    
for index, row in scores.iterrows():
	index_predict = index + '_predict'
	tp, scores.loc[index,'perc_true_pos'] = score(index,index_predict,1,1)
	fp, scores.loc[index,'perc_true_neg'] = score(index,index_predict,0,0)
	fp, scores.loc[index,'perc_fals_pos'] = score(index,index_predict,1,0)
	fn, scores.loc[index,'perc_fals_neg'] = score(index,index_predict,0,1)
	scores2.loc[index,'precision'] = precision(tp,fp)
	scores2.loc[index,'recall'   ] = recall   (tp,fn)

print(scores2.head(7))