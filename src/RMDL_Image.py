import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.metrics import accuracy_score
from keras.datasets import cifar,mnist
import numpy as np
import Plot
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import collections
from keras.models import Sequential
from sklearn.metrics import f1_score,precision_recall_fscore_support
import BuildModel
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from keras.callbacks import ModelCheckpoint
np.random.seed(7)

def image_classifciation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                            n_epochs):
    y_proba = []
    model = []
    score = []
    history_ = []
    number_of_classes = np.max(y_train)+1
    #X_train, y_train, X_test, y_test, shape, number_of_classes = Image_Data_load.load(Data_Image)
    for i in range(0, Random_Deep[0]):
        print("MNIST DNN ", i, "\n")
        model.append(Sequential())
        model[i], model_tmp = BuildModel.buildModel_DNN_image(shape, number_of_classes, 0)
        filepath = "weights\weights_DNN_" + str(i) + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        history = model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                               epochs=n_epochs[0], batch_size=batch_size, callbacks=callbacks_list,
                               verbose=2)
        history_.append(history)
        model_tmp.load_weights(filepath)

        if sparse_categorical == 0:
            model_tmp.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
        else:
            model_tmp.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

        y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
        y_proba.append(np.array(y_pr))
        score.append(accuracy_score(y_test, y_pr))

    for i in range(0, Random_Deep[2]):
        print("RNN ", i, "\n")
        model.append(Sequential())
        model[i], model_tmp = BuildModel.Image_model_RNN(number_of_classes, shape)

        filepath = "weights\weights_RNN_" + str(i) + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        history = model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                               epochs=n_epochs[2], batch_size=batch_size, verbose=2, callbacks=callbacks_list)
        model_tmp.load_weights(filepath)
        model_tmp.compile(loss='sparse_categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        history_.append(history)

        y_pr = model_tmp.predict(X_test, batch_size=batch_size)
        y_pr = np.argmax(y_pr, axis=1)
        y_proba.append(np.array(y_pr))
        score.append(accuracy_score(y_test, y_pr))

    # reshape to be [samples][pixels][width][height]
    i=0
    while i < Random_Deep[1]:
        try:
            print("CNN ", i, "\n")
            model.append(Sequential())
            model[i], model_tmp = BuildModel.Image_model_CNN(number_of_classes, shape)

            filepath = "weights\weights_CNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                                   epochs=n_epochs[1], batch_size=batch_size, callbacks=callbacks_list, verbose=2)
            history_.append(history)
            model_tmp.load_weights(filepath)
            model_tmp.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                              metrics=['accuracy'])
            y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
            i = i+1
        except:
            print("Error in model ", i, "   try to re-generate an other model")


    y_proba = np.array(y_proba).transpose()
    print(y_proba.shape)
    final_y = []
    for i in range(0, y_proba.shape[0]):
        a = np.array(y_proba[i, :])
        a = collections.Counter(a).most_common()[0][0]
        final_y.append(a)
    F_score = accuracy_score(y_test, final_y)
    F1 = f1_score(y_test, final_y, average='micro')
    F2 = f1_score(y_test, final_y, average='macro')
    F3 = f1_score(y_test, final_y, average='weighted')
    cnf_matrix = confusion_matrix(y_test, final_y)
    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    classes = list(range(0,np.max(y_test)))
    Plot.plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    print(score)
    print(F_score)
    print(F1)
    print(F2)
    print(F3)
