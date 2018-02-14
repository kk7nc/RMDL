import numpy as np
from keras.datasets import cifar10,cifar100,mnist
import scipy.io as sio

def load(Data_Image):
    if Data_Image==1:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train_D = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
        X_test_D = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')
        X_train = X_train_D / 255.0
        X_test = X_test_D / 255.0
        number_of_classes = np.unique(y_train).shape[0]
        shape = (28, 28,1)
    elif Data_Image==2:
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # fetch CIFAR-100 data
            X_train_D = X_train.astype('float32')
            X_test_D = X_test.astype('float32')
            X_train = X_train_D / 255.0
            X_test = X_test_D / 255.0
            number_of_classes = np.unique(y_train).shape[0]  # there are 100 image classes
            shape = (32, 32, 3)
    elif Data_Image==3:
            train_data = sio.loadmat("D:\\Google\\RDeeps\\DATA\image\\train_32x32.mat")
            test_data = sio.loadmat("D:\\Google\\RDeeps\DATA\\image\\test_32x32.mat")

            # access to the dict
            X_tr = train_data['X']
            y_train = train_data['y']
            X_te = test_data['X']
            y_test = test_data['y']
            x_train = []
            for i in range(X_tr.shape[3]):
                x_train.append(X_tr[:, :, :, i])
            x_test = []
            LLL = np.array(X_te[:, :, :, 234])
            for i in range(X_te.shape[3]):
                x_test.append(X_te[:, :, :, i])
            X_train = np.array(x_train)
            X_test = np.array(x_test)
            X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
            X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
            X_train = X_train / 256
            X_test = X_test / 256
            y_train -= 1
            y_test -= 1
            print(np.max(y_train), np.min(y_train))
            number_of_classes = np.unique(y_train).shape[0]

            print("number of classes:  " + str(number_of_classes))
            shape = (32, 32, 3)

    return (X_train, y_train, X_test, y_test,shape, number_of_classes)