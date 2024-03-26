

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import random as rd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# %matplotlib inline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Code from scikit-learn
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


from sklearn.pipeline import make_pipeline


#### LOAD DATA

file_name = '/Users/francoisramon/Desktop/These/Catatonia/volume_catacof_maj.xlsx'
features = '/Users/francoisramon/Desktop/These/Catatonia/results.csv'


df = pd.read_excel(file_name) # reading data
feats = pd.read_csv(features)


########################################
####         PRE preprocessing       ###
########################################


# Add features from demographic and volumic. Use One Hot encorder for categorical data
feat_list = ["statut ","Sex","Age","ECT"]
feat_volumes_list = feats.iloc[1:, 1].to_list()

full_feat_list = feat_list + feat_volumes_list
df_feat = df[full_feat_list]
df_feat['Sex'] = df_feat['Sex'].astype('category')
df_feat['Sex_new'] = df_feat['Sex'].cat.codes
enc = OneHotEncoder()
enc_data = pd.DataFrame(enc.fit_transform(
    df_feat[['Sex_new']]).toarray())
df_feat = df_feat.join(enc_data)


## Read data
y = df_feat["statut "].values # 1 for Melanoma and 0 for healthy
class_names = ["no catatonia","catatonia"]
feats_for_x = ["Sex_new","Age","ECT"] + feat_volumes_list
X = df_feat[feats_for_x]

###################################
###     Dataset preparation     ###
###################################

# Shuffle data randomly

seed = 42
rd.seed(seed)
Xp=shuffle(X, random_state = seed)
yp=shuffle(y,random_state = seed)


#### Create training and test set #####
X_train, X_test, y_train, y_test = train_test_split(Xp, yp, test_size=0.3, random_state=42,stratify=yp)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scale=scaler.transform(X_train)
X_test_scale=scaler.transform(X_test)


####################################
###     Model fitting           ####
####################################

# Fitting LDA
print("Fitting LDA to training set")
t0 = time()
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scale, y_train)
y_pred = lda.predict(X_test_scale)
print(classification_report(y_test, y_pred))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='LDA Normalized confusion matrix')
plt.show()

lda_score = cross_val_score(lda,X=Xp, y=np.ravel(yp),cv=5)
print(" Average and std CV score : {0} +- {1}".format(lda_score.mean(), lda_score.std() ))




