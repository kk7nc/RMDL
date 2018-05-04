'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification

 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 * Last Update: May 3rd, 2018
 * This file is part of  RMDL project, University of Virginia.
 * Free to use, change, share and distribute source code of RMDL
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from keras.datasets import cifar10
from RMDL import RMDL_Image as RMDL

if __name__ == "__main__":
    number_of_classes = 40
    shape = (64, 64, 1)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    batch_size = 100
    sparse_categorical = 0
    n_epochs = [500, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 0, 3]  ## DNN--RNN-CNN

    RMDL.Image_Classification(x_train, y_train, x_test, y_test,(32,32,3),
                              batch_size=batch_size,random_deep=Random_Deep,
                              epochs=n_epochs)
