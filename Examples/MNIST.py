import sys
sys.path.append('../src')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from keras.datasets import mnist
import numpy as np
import RMDL

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test_D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_train = X_train_D / 255.0
    X_test = X_test_D / 255.0
    number_of_classes = np.unique(y_train).shape[0]
    shape = (28, 28, 1)
    batch_size = 100
    sparse_categorical = 0
    n_epochs = [5000, 500, 1000]  ## DNN--RNN-CNN
    Random_Deep = [0, 30, 0]  ## DNN--RNN-CNN
    RMDL.Image_Classifcation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                            n_epochs)