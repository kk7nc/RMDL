"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: kk7nc@virginia.edu

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from RMDL import RMDL_Image as RMDL

if __name__ == "__main__":
    number_of_classes = 40
    shape = (64, 64, 1)
    data = fetch_olivetti_faces()
    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                    data.target, stratify=data.target, test_size=40)
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 64, 64, 1).astype('float32')

    batch_size = 100
    sparse_categorical = 0
    n_epochs = [150, 150, 150]  ## DNN--RNN-CNN
    Random_Deep = [0, 0, 3]  ## DNN--RNN-CNN
    RMDL.Image_Classification(X_train, y_train, X_test, y_test,
                              shape,
                              random_optimizor=False,
                              batch_size=batch_size,
                              random_deep=Random_Deep,
                              epochs=n_epochs)