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
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''



import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import RMDL_Text as RMDL
import sys
sys.path.append('../Download_datasets')
from sklearn.cross_validation import train_test_split, cross_val_score
import text_feature_extraction as txt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    file_x = "..\Data\X.csv"
    file_y = "..\Data\Y.csv"
    content = pd.read_csv(file_x, encoding="utf-8")
    Label = pd.read_csv(file_y, encoding="utf-8")
    # content = content.as_matrix()
    content = content.ix[:, 1]
    content = np.array(content).ravel()
    print(np.array(content).transpose().shape)
    Label = Label.as_matrix()
    Label = np.matrix(Label)
    np.random.seed(7)
    # print(Label)
    content = [txt.text_cleaner(x) for x in content]
    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.33, random_state=1)
    batch_size = 64
    sparse_categorical = 0
    n_epochs = [5000, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)