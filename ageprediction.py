import os
import numpy as np
import pandas as pd
from skimage.io import imread
from time import time
import matplotlib.pyplot as plt
import random as rd

from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeClassifier,Ridge
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Load data from the Excel file
data = pd.read_csv("/Users/francoisramon/Desktop/These/Catatonia/full_catacof.csv")

###################################
###     Dataset preparation     ###
###################################

y = data["Age"]

X = data.drop(columns=["Age"])

# Handle categorical data in columns "ORB_L" and "ORB_R"
X = pd.get_dummies(X, columns=["Sex","type_OFC_G", "type_OFC_D"])
print(X.columns)
data_shuffled = data.sample(frac=1, random_state=42)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models
models = {
  
    'Ridge': Ridge(max_iter=10000)

}

# Define parameter grids for hyperparameter tuning
param_grids = {
   
    'Ridge': {'regressor__alpha': [0.01,1,10,100,1000]}
}


# Perform nested cross-validation and evaluate models
results = {}
for model_name, model in models.items():
    #lda = LinearDiscriminantAnalysis()
    pipeline = Pipeline(steps = [
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])

    # Define the inner cross-validation to select hyperparameters
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Perform hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=inner_cv, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model from hyperparameter tuning
    best_model = grid_search.best_estimator_
    
    # Perform outer cross-validation to evaluate the model
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=outer_cv, scoring='neg_mean_absolute_error')
    
    # Evaluate the model on test data
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)


    plt.plot(y_pred,y_test,"bo")
    plt.xlabel("y pred")
    plt.ylabel("y test")
    plt.show()
    r2 =  r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    print(f"r2 {r2}")
    print(f"mae {mae}")

    #auc = roc_auc_score(y_test, best_model.predict_proba(preprocessing.transform(X_test_standardized))[:, 1])
    #acc = accuracy_score(y_test,y_pred)
    # Store the results
    results[model_name] = {
        #'cm': confusion_matrix(y_test, y_pred),
        'Best_Params': grid_search.best_params_,
        'CV_mae': np.mean(cv_scores),
        'CV_stdmae': np.std(cv_scores)
        #'Test_BACC': bacc,
        #'Test_ACC':acc
        #'Test_AUC': auc
    }
   # Plot confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm/np.sum(cm, axis=1)[:, np.newaxis], display_labels=["No Catatonia", "Catatonia"])
    # disp.plot(cmap='Blues', values_format='.2f')
    # plt.title(f'Confusion Matrix - {model_name}')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.show()


# # Print the results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Best Parameters: {result['Best_Params']}")
    print(f"Cross-Validation MAE: {result['CV_mae']} +-  {result['CV_stdmae']} ")
   # print(f"Test Balanced Accuracy: {result['Test_BACC']}")










