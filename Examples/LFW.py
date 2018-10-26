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

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
from RMDL import RMDL_Image as RMDL

if __name__ == "__main__":

    shape = (125, 94, 1)

    lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=1.0)

    # introspect the images arrays to find the shapes (for plotting)
    print(lfw_people.images.shape)

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]
    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    # #############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)


    X_train = X_train.reshape(X_train.shape[0], 125, 94, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 125, 94, 1).astype('float32')
    number_of_classes = np.max(y_train)+1
    X_train /= 255
    X_test /= 255
    batch_size = 100
    sparse_categorical = 0
    n_epochs = [5000, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN
    RMDL.Image_Classification(X_train, y_train, X_test, y_test,
                              shape,
                              batch_size=batch_size,
                              random_deep=Random_Deep,
                              epochs=n_epochs)