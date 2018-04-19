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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.metrics import accuracy_score
import numpy as np
from RMDL import Plot as Plot
import gc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import collections
from sklearn.metrics import f1_score
from RMDL import BuildModel as BuildModel
from RMDL import Global as G
from keras.callbacks import ModelCheckpoint
np.random.seed(7)

def Image_Classification(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                            n_epochs,plot=True):
    G.setup(text=False)
    y_proba = []

    score = []
    history_ = []
    number_of_classes = np.max(y_train)+1
    #X_train, y_train, X_test, y_test, shape, number_of_classes = Image_Data_load.load(Data_Image)
    i =0
    while i < Random_Deep[0]:
        try:
            print("DNN ", i, "\n")
            model_DNN, model_tmp = BuildModel.buildModel_DNN_image(shape, number_of_classes, 0)
            filepath = "weights\weights_DNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model_DNN.fit(X_train, y_train, validation_data=(X_test, y_test),
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
            i = i + 1
            del model_tmp
            del model_DNN
            gc.collect()
        except:
            print("Error in model ", i, "   try to re-generate an other model")
    i =0
    while i < Random_Deep[1]:
        try:
            print("RNN ", i, "\n")
            model_RNN, model_tmp = BuildModel.Image_model_RNN(number_of_classes, shape)

            filepath = "weights\weights_RNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model_RNN.fit(X_train, y_train, validation_data=(X_test, y_test),
                                   epochs=n_epochs[1], batch_size=batch_size, verbose=2, callbacks=callbacks_list)
            model_tmp.load_weights(filepath)
            model_tmp.compile(loss='sparse_categorical_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
            history_.append(history)

            y_pr = model_tmp.predict(X_test, batch_size=batch_size)
            y_pr = np.argmax(y_pr, axis=1)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
            i = i+1
            del model_tmp
            del model_RNN
            gc.collect()
        except:
            print("Error in model ", i, "   try to re-generate an other model")

    # reshape to be [samples][pixels][width][height]
    i=0
    while i < Random_Deep[2]:

        print("CNN ", i, "\n")
        model_CNN, model_tmp = BuildModel.Image_model_CNN(number_of_classes, shape)

        filepath = "weights\weights_CNN_" + str(i) + ".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='max')
        callbacks_list = [checkpoint]

        history = model_CNN.fit(X_train, y_train, validation_data=(X_test, y_test),
                               epochs=n_epochs[2], batch_size=batch_size, callbacks=callbacks_list, verbose=2)
        history_.append(history)
        model_tmp.load_weights(filepath)
        model_tmp.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                          metrics=['accuracy'])
        y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
        y_proba.append(np.array(y_pr))
        score.append(accuracy_score(y_test, y_pr))
        i = i+1
        del model_tmp
        del model_CNN
        gc.collect()



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
    if plot:
        # Plot non-normalized confusion matrix
        classes = list(range(0,np.max(y_test)+1))
        Plot.plot_confusion_matrix(cnf_matrix, classes=classes,
                         title='Confusion matrix, without normalization')
        Plot.plot_confusion_matrix(cnf_matrix, classes=classes,normalize=True,
                              title='Confusion matrix, without normalization')

    if plot:
        Plot.RMDL_epoch(history_)

    print(score)
    print(F_score)
    print(F1)
    print(F2)
    print(F3)
