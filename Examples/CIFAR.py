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
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import src.RMDL_Image as RMDL

if __name__ == "__main__":
    number_of_classes = 40
    shape = (64, 64, 1)

    data = fetch_olivetti_faces()

    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.target, stratify=data.target, test_size=200)
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1).astype('float32')
    batch_size = 100
    sparse_categorical = 0
    n_epochs = [5000, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN

    RMDL.Image_Classification(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                            n_epochs)