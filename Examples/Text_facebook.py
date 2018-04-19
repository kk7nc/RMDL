'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification

 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 *
 * This file is part of  RMDL project, University of Virginia.
 *
 * Free to use, change, share and distribute source code of RMDL
 *
 *
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 *
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * 
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,1"
from RMDL import RMDL_Text as RMDL
import sys
sys.path.append('../Download_datasets')
from sklearn.cross_validation import train_test_split, cross_val_score
from RMDL import text_feature_extraction as txt
import numpy as np
import pandas as pd


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


if __name__ == "__main__":

    for i in range(0,32):
        print("Start %d",i)
        file = "D:\CHI\Facebook\out_final.csv"
        df = pd.read_csv(file, encoding="utf-8")
        PID = df['ID']
        PID = np.unique(PID)
        df_train = df.ix[df['ID'] != PID[i]]
        df_test = df.ix[df['ID'] == PID[i]]
        X_train = df_train['M']
        X_test = df_test['M']
        X_train = np.array(X_train).astype(str)
        X_test = np.array(X_test).astype(str)

        y_train = df_train['e']
        y_test = df_test['e']
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        y_train[y_train == 'Attempt'] = 0
        y_train[y_train == 'Ideation'] = 0
        y_train[y_train == 'Depression'] = 1
        y_train[y_train == 'Positive'] = 1

        y_test[y_test == 'Attempt'] = 0
        y_test[y_test == 'Ideation'] = 0
        y_test[y_test == 'Depression'] = 1
        y_test[y_test == 'Positive'] = 1





        # content = content.as_matrix()

        X_train = [txt.text_cleaner(x, deep_clean=True) for x in X_train]
        X_test = [txt.text_cleaner(x, deep_clean=True) for x in X_test]


        #X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.1, random_state=42)
        batch_size = 256
        sparse_categorical = 0
        n_epochs = [10, 100, 50]  ## DNN--RNN-CNN
        Random_Deep = [1, 0, 0]  ## DNN--RNN-CNN
        #X_train = np.matrix(X_train)
        y_train = np.matrix(y_train).transpose().astype(int)
        #X_test = np.matrix(X_test)
        y_test = np.matrix(y_test).transpose().astype(int)

        RMDL.Text_Classification(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                                n_epochs)
        clear_all()