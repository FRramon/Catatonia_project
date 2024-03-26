# -*- coding: utf-8 -*-
"""
Created on Fri March 15 2024

@author: francois
"""

import os
import sys
import time
import glob
import re
import copy
import pickle
import shutil
import json
import subprocess

import numpy as np
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import nibabel
import nilearn.image
from nilearn.image import resample_to_img
import nilearn.image
from nilearn.image import new_img_like
from nilearn import plotting

sys.path.append('/Users/francoisramon/nitk')
sys.path.append('/home/francoisramon//pylearn-mulm')
sys.path.append('/home/francoisramon//pylearn-parsimony')

from nitk.utils import maps_similarity, arr_threshold_from_norm2_ratio, arr_clusters
from nitk.image import img_to_array, global_scaling, compute_brain_mask, rm_small_clusters, vec_to_niimg, plot_glass_brains
#, img_plot_glass_brain
#from nitk.stats import Residualizer
from mulm.residualizer import Residualizer
from nitk.mapreduce import dict_product, MapReduce, reduce_cv_classif

import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sklearn.metrics as metrics
from statannot.statannot import add_stat_annotation
import statsmodels.api as sm
from scipy.stats import ttest_ind, pearsonr, mannwhitneyu
import random


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay

from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline


######## STEPS ###########

###### ELASTIC NET  (++)
###### PERMUTATION FOR SIGNIFICANCE (++ )
###### REGION IMPORTANCE (++)
###### PEARSON R FOR INDIVIDUAL REGIONS
###### MATURATIONAL CHARTS
###### COMPUTATION /GRAPHS/ RESUTLS
##### REGRESSION WITH CONFOUNDERS : AGE, CANNABIS, OUTCOME, MEDICATION (++)

######################### LOAD DATA ##############################


def fit_model(data,residualize,residualizer):

	listcols = data.columns
	cols_to_keep = [c for c in listcols if not "thickness" in c and not "asymmetry" in c and not "%" in c and not "SNR" in c and not "Scale Factor" in c and not "N_anomymisation" in c and not "tesla" in c ]
	print(cols_to_keep)
	data = data[cols_to_keep]
	# Y is the factor to predict : catatonia :1, no catatonia:0
	y = data["statut"]
	# Z is the dataframe containing demographical data + confounds like medication, psychiatric care duration
	Z = data[["Age","Sex","duree_TT"]]#,"equivalent_olz_1","equivalent_valium_10"]]
	# X is the features dataframe
	X = data.drop(columns=["statut","Age","Sex","duree_TT"])#,"equivalent_olz_1","equivalent_valium_10"])

	# Handle categorical data in columns "ORB_L" and "ORB_R" in X, and "Sex" in Z
	#X = pd.get_dummies(X, columns=["type_OFC_G", "type_OFC_D"])
	Z = pd.get_dummies(Z, columns=["Sex"])

	#Split the data into train and test sets
	X_train, X_test, y_train, y_test,Z_train,Z_test = train_test_split(X, y, Z, test_size=0.3, random_state=42)


	# X_train = X_train.to_numpy()
	# Z_train = Z_train.to_numpy()
	# y_train = y_train.to_numpy()

	# X_test = X_test.to_numpy()
	# Z_test = Z_test.to_numpy()
	# y_test = y_test.to_numpy()

	#contrast_res = np.ones(Z_train.shape[1]).astype(bool)

#	print(contrast_res.shape[0] == Z_train.shape[1])


	print(type(X_train))

	# print(Z_train.shape)
	print(type(Z_train))

	if residualize == 'yes':
		#data.to_csv("/Users/francoisramon/Desktop/These/Catatonia/volumes_df_15_03.csv")


		cols_vols = [c for c in cols_to_keep if "cm3" in c]
		for col in cols_vols:
			print(col)
			colXtrain = X_train[col].to_numpy()
			colXtest = X_test[col].to_numpy()
			X_train[col] = residualizer.fit_transform(colXtrain[:,None], residualizer.get_design_mat(X_train))
			X_test[col] = residualizer.fit_transform(colXtest[:,None], residualizer.get_design_mat(X_train))



	    # residualizer.fit(X_train, Z_train)
	    # X_train = residualizer.transform(X_train, Z_train)
	    # X_test = residualizer.transform(X_test, Z_test)
	# elif residualize == 'biased':
	#         residualizer.fit(X, Z)
	#         X_train = residualizer.transform(X_train, Z_train)
	#         X_test = residualizer.transform(X_test, Z_test)


	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)



# key_values_input = dict_product(estimators_dict, dict(resdualizeYes="yes"), cv_dict,
#         {'Xim_%s' % dataset :data['Xim']}, {'y_%s' % dataset :data['y']},
#         {'Z_%s' % dataset :data['Z']}, {'Xdemoclin_%s' % dataset :data['Xdemoclin']},
#         {'residualizer_%s' % dataset :data['residualizer']})



data = pd.read_excel("/Users/francoisramon/Downloads/classif.xlsx")
# listcols = data.columns
# cols_to_keep = [c for c in listcols if not "thickness" in c and not "asymmetry" in c and not "%" in c and not "SNR" in c and not "Scale Factor" in c and not "N_anomymisation" in c and not "tesla" in c ]
# print(cols_to_keep)
# data = data[cols_to_keep]
#data_shuffled = data.sample(frac=1, random_state=42)
residualize = 'yes'
formula_res = "Age + Sex + duree_TT"#, "Age + Sex + duree_TT + statut"  
residualizer = Residualizer(data=data, formula_res=formula_res)#, formula_full=formula_full)

# X = data.drop(columns=["statut","Age","Sex","duree_pec_psy","equivalent_olz_1","equivalent_valium_10"])
# Xres = residualizer.fit_transform(X, residualizer.get_design_mat(data))
# print(Xres)
fit_model(data,residualize,residualizer)


# def fit_predict(key, estimator_img, residualize, split, Xim, y, Z, Xdemoclin, residualizer):


#   #  estimator_img = copy.deepcopy(estimator_img)
#     train, test = split
#     print("fit_predict", Xim.shape, Xdemoclin.shape, Z.shape, y.shape, np.max(train), np.max(test))
#     Xim_train, Xim_test, Xdemoclin_train, Xdemoclin_test, Z_train, Z_test, y_train =\
#         Xim[train, :], Xim[test, :], Xdemoclin[train, :], Xdemoclin[test, :], Z[train, :], Z[test, :], y[train]

#     # Images based predictor
#     # Residualization
#     if residualize == 'yes':
#         residualizer.fit(Xim_train, Z_train)
#         Xim_train = residualizer.transform(Xim_train, Z_train)
#         Xim_test = residualizer.transform(Xim_test, Z_test)

#     elif residualize == 'biased':
#         residualizer.fit(Xim, Z)
#         Xim_train = residualizer.transform(Xim_train, Z_train)
#         Xim_test = residualizer.transform(Xim_test, Z_test)

#     elif residualize == 'no':
#         pass

#     scaler = StandardScaler()
#     Xim_train = scaler.fit_transform(Xim_train)
#     Xim_test = scaler.transform(Xim_test)

#     estimator_img.fit(Xim_train, y_train)

#     y_test_img = estimator_img.predict(Xim_test)
    
#     if hasattr(estimator_img, 'decision_function'):
#         score_test_img = estimator_img.decision_function(Xim_test)
#         score_train_img = estimator_img.decision_function(Xim_train)
#     elif hasattr(estimator_img, 'predict_log_proba'):
#         score_test_img = estimator_img.predict_log_proba(Xim_test)[:, 1]
#         score_train_img = estimator_img.predict_log_proba(Xim_train)[:, 1]
#     elif hasattr(estimator_img, 'predict_proba'):
#         score_test_img = estimator_img.predict_proba(Xim_test)[:, 1]
#         score_train_img = estimator_img.predict_proba(Xim_train)[:, 1]
#     else:
#         raise AttributeError("Missing: decision_function or predict_log_proba or predict_proba")

#     # Demographic/clinic based predictor
#     estimator_democlin = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=False)
#     scaler = StandardScaler()
#     Xdemoclin_train = scaler.fit_transform(Xdemoclin_train)
#     Xdemoclin_test = scaler.transform(Xdemoclin_test)
#     estimator_democlin.fit(Xdemoclin_train, y_train)
#     y_test_democlin = estimator_democlin.predict(Xdemoclin_test)
#     score_test_democlin = estimator_democlin.decision_function(Xdemoclin_test)
#     score_train_democlin = estimator_democlin.decision_function(Xdemoclin_train)

#     # STACK DEMO + IMG
#     estimator_stck = lm.LogisticRegression(C=1, class_weight='balanced', fit_intercept=True)

#     Xstck_train = np.c_[score_train_democlin, score_train_img]
#     Xstck_test = np.c_[score_test_democlin, score_test_img]
#     scaler = StandardScaler()
#     Xstck_train = scaler.fit_transform(Xstck_train)
#     Xstck_test = scaler.transform(Xstck_test)
#     estimator_stck.fit(Xstck_train, y_train)

#     y_test_stck = estimator_stck.predict(Xstck_test)
#     score_test_stck = estimator_stck.predict_log_proba(Xstck_test)[:, 1]
#     score_train_stck = estimator_stck.predict_log_proba(Xstck_train)[:, 1]

#     # Retrieve coeficient if possible
#     coef_img = None

#     if hasattr(estimator_img, 'best_estimator_'):  # GridSearch case
#         estimator_img = estimator_img.best_estimator_

#     if hasattr(estimator_img, 'coef_'):
#         coef_img = estimator_img.coef_

#     return dict(y_test_img=y_test_img, score_test_img=score_test_img,
#                 y_test_democlin=y_test_democlin, score_test_democlin=score_test_democlin,
#                 y_test_stck=y_test_stck, score_test_stck=score_test_stck,
#                 coef_img=coef_img)






# start_time = time.time()
#     mp = MapReduce(n_jobs=NJOBS, shared_dir=mapreduce_sharedir, pass_key=True, verbose=20)
#     mp.map(fit_predict, key_values_input)
#     key_vals_output = mp.reduce_collect_outputs()


