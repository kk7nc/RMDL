import sys
sys.path.append('../src')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import numpy as np
import RMDL

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
    n_epochs = [5000, 500, 1000]  ## DNN--RNN-CNN
    Random_Deep = [0, 30, 0]  ## DNN--RNN-CNN

    RMDL.Image_Classifcation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                            n_epochs)