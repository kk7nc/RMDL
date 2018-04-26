'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification

 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 * Last Update: 04/25/2018
 * This file is part of  RMDL project, University of Virginia.
 * Free to use, change, share and distribute source code of RMDL
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from sklearn.metrics import accuracy_score
import numpy as np
from RMDL import Plot as Plot
import gc
from sklearn.metrics import confusion_matrix
import collections
from sklearn.metrics import f1_score
from RMDL import BuildModel as BuildModel
from RMDL import Global as G
from keras.callbacks import ModelCheckpoint
np.random.seed(7)


def Image_Classification(x_train, y_train, x_test, y_test, shape, batch_size=128,
                         sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=True,
                         min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                         min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32, max_nodes_rnn=128,
                         min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                         random_state=42, random_optimizor=True, dropout=0.05):
    np.random.seed(random_state)
    G.setup(text=False)
    y_proba = []

    score = []
    history_ = []
    if sparse_categorical:
        number_of_classes = np.max(y_train)+1
    else:
        number_of_classes = np.shape(y_train)[0]

    i =0
    while i < random_deep[0]:
        try:
            print("DNN ", i, "\n")
            model_DNN, model_tmp = BuildModel.Build_Model_DNN_Image(shape,
                                                                    number_of_classes,
                                                                    sparse_categorical,
                                                                    min_hidden_layer_dnn,
                                                                    max_hidden_layer_dnn,
                                                                    min_nodes_dnn,
                                                                    max_nodes_dnn,
                                                                    random_optimizor,
                                                                    dropout)


            filepath = "weights\weights_DNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model_DNN.fit(x_train, y_train,
                                    validation_data=(x_test, y_test),
                                    epochs=epochs[0],
                                    batch_size=batch_size,
                                    callbacks=callbacks_list,
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

            y_pr = model_tmp.predict_classes(x_test, batch_size=batch_size)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
            i = i + 1
            del model_tmp
            del model_DNN
            gc.collect()
        except:
            print("Error in model", i, "try to re-generate an other model")
            if max_hidden_layer_dnn > 3:
                max_hidden_layer_dnn -= 1
            if max_nodes_dnn > 256:
                max_nodes_dnn -= 8


    i =0
    while i < random_deep[1]:
        try:
            print("RNN ", i, "\n")
            model_RNN, model_tmp = BuildModel.Build_Model_RNN_Image(shape,
                                                                    number_of_classes,
                                                                    sparse_categorical,
                                                                    min_nodes_rnn,
                                                                    max_nodes_rnn,
                                                                    random_optimizor,
                                                                    dropout)

            filepath = "weights\weights_RNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model_RNN.fit(x_train, y_train,
                                    validation_data=(x_test, y_test),
                                    epochs=epochs[1],
                                    batch_size=batch_size,
                                    verbose=2,
                                    callbacks=callbacks_list)

            model_tmp.load_weights(filepath)
            model_tmp.compile(loss='sparse_categorical_crossentropy',
                              optimizer='rmsprop',
                              metrics=['accuracy'])
            history_.append(history)

            y_pr = model_tmp.predict(x_test, batch_size=batch_size)
            y_pr = np.argmax(y_pr, axis=1)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
            i = i+1
            del model_tmp
            del model_RNN
            gc.collect()
        except:
            print("Error in model", i, " try to re-generate another model")
            if max_hidden_layer_rnn > 3:
                max_hidden_layer_rnn -= 1
            if max_nodes_rnn > 64:
                max_nodes_rnn -= 2

    # reshape to be [samples][pixels][width][height]
    i=0
    while i < random_deep[2]:
        try:
            print("CNN ", i, "\n")
            model_CNN, model_tmp = BuildModel.Build_Model_CNN_Image(shape,
                                                                    number_of_classes,
                                                                    sparse_categorical,
                                                                    min_hidden_layer_cnn,
                                                                    max_hidden_layer_cnn,
                                                                    min_nodes_cnn,
                                                                    max_nodes_cnn,
                                                                    random_optimizor,
                                                                    dropout)

            filepath = "weights\weights_CNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            history = model_CNN.fit(x_train, y_train,
                                    validation_data=(x_test, y_test),
                                    epochs=epochs[2],
                                    batch_size=batch_size,
                                    callbacks=callbacks_list,
                                    verbose=2)
            history_.append(history)
            model_tmp.load_weights(filepath)
            model_tmp.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

            y_pr = model_tmp.predict_classes(x_test, batch_size=batch_size)
            y_proba.append(np.array(y_pr))
            score.append(accuracy_score(y_test, y_pr))
            i = i+1
            del model_tmp
            del model_CNN
            gc.collect()
        except:
            print("Error in model", i, " try to re-generate another model")
            if max_hidden_layer_cnn > 5:
                max_hidden_layer_cnn -= 1
            if max_nodes_cnn > 128:
                max_nodes_cnn -= 2
                min_nodes_cnn -= 1



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

    print(y_proba.shape)
    print("Accuracy of",len(score),"models:",score)
    print("Accuracy:",F_score)
    print("F1_Micro:",F1)
    print("F1_Macro:",F2)
    print("F1_weighted:",F3)
