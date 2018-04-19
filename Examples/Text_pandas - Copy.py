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

from RMDL import RMDL_Text as RMDL
from sklearn.cross_validation import train_test_split, cross_val_score
from RMDL import text_feature_extraction as txt
import numpy as np
import pandas as pd
from sklearn import preprocessing

if __name__ == "__main__":
    file = "C:/Users/kamran/Downloads/accidents01-17.csv"
    df = pd.read_csv(file, encoding="utf-8")
    content = df['naritive']
    print(np.array(content).transpose().shape)
    Label = df['cause_letter']
    le = preprocessing.LabelEncoder()
    Label = np.array(Label).astype(str)
    Target = np.unique(Label)
    le.fit(Target)
    Label = le.transform(Label)
    print(Label)
    print(Target)
    np.random.seed(7)
    content = np.array(content).astype(str)
    # print(Label)
    content = [txt.text_cleaner(x,deep_clean=True) for x in content]
    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=42)
    batch_size = 256
    sparse_categorical = 0
    n_epochs = [20, 15, 20]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN
    RMDL.Text_Classification(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)