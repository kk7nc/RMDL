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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
                         sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
                         min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                         max_hidden_layer_rnn=5, min_nodes_rnn=32, max_nodes_rnn=128,
                         min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                         random_state=42, random_optimizor=True, dropout=0.05):
    """
    def Image_Classification(x_train, y_train, x_test, y_test, shape, batch_size=128,
                             sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500], plot=False,
                             min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                             min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32, max_nodes_rnn=128,
                             min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                             random_state=42, random_optimizor=True, dropout=0.05):

            Parameters
            ----------
                x_train : string
                    input X for training
                y_train : int
                    input Y for training
                x_test : string
                    input X for testing
                x_test : int
                    input Y for testing
                shape : np.shape
                    shape of image. The most common situation would be a 2D input with shape (batch_size, input_dim).
                batch_size : Integer, , optional
                    Number of samples per gradient update. If unspecified, it will default to 128
                MAX_NB_WORDS: int, optional
                    Maximum number of unique words in datasets, it will default to 75000.
                GloVe_dir: String, optional
                    Address of GloVe or any pre-trained directory, it will default to null which glove.6B.zip will be download.
                GloVe_dir: String, optional
                    Which version of GloVe or pre-trained word emending will be used, it will default to glove.6B.50d.txt.
                    NOTE: if you use other version of GloVe EMBEDDING_DIM must be same dimensions.
                sparse_categorical: bool.
                    When target's dataset is (n,1) should be True, it will default to True.
                random_deep: array of int [3], optional
                    Number of ensembled model used in RMDL random_deep[0] is number of DNN, random_deep[1] is number of RNN, random_deep[0] is number of CNN, it will default to [3, 3, 3].
                epochs: array of int [3], optional
                    Number of epochs in each ensembled model used in RMDL epochs[0] is number of epochs used in DNN, epochs[1] is number of epochs used in RNN, epochs[0] is number of epochs used in CNN, it will default to [500, 500, 500].
                plot: bool, optional
                    True: shows confusion matrix and accuracy and loss
                min_hidden_layer_dnn: Integer, optional
                    Lower Bounds of hidden layers of DNN used in RMDL, it will default to 1.
                max_hidden_layer_dnn: Integer, optional
                    Upper bounds of hidden layers of DNN used in RMDL, it will default to 8.
                min_nodes_dnn: Integer, optional
                    Lower bounds of nodes in each layer of DNN used in RMDL, it will default to 128.
                max_nodes_dnn: Integer, optional
                    Upper bounds of nodes in each layer of DNN used in RMDL, it will default to 1024.
                min_hidden_layer_rnn: Integer, optional
                    Lower Bounds of hidden layers of RNN used in RMDL, it will default to 1.
                min_hidden_layer_rnn: Integer, optional
                    Upper Bounds of hidden layers of RNN used in RMDL, it will default to 5.
                min_nodes_rnn: Integer, optional
                    Lower bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL, it will default to 32.
                max_nodes_rnn: Integer, optional
                    Upper bounds of nodes (LSTM or GRU) in each layer of RNN used in RMDL, it will default to 128.
                min_hidden_layer_cnn: Integer, optional
                    Lower Bounds of hidden layers of CNN used in RMDL, it will default to 3.
                max_hidden_layer_cnn: Integer, optional
                    Upper Bounds of hidden layers of CNN used in RMDL, it will default to 10.
                min_nodes_cnn: Integer, optional
                    Lower bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 128.
                min_nodes_cnn: Integer, optional
                    Upper bounds of nodes (2D convolution layer) in each layer of CNN used in RMDL, it will default to 512.
                random_state : Integer, optional
                    RandomState instance or None, optional (default=None)
                    If Integer, random_state is the seed used by the random number generator;
                random_optimizor : bool, optional
                    If False, all models use adam optimizer. If True, all models use random optimizers. it will default to True
                dropout: Float, optional
                    between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.

        """

    if len(x_train) != len(y_train):
        raise ValueError('shape of x_train and y_train must be equal'
                         'The x_train has ' + str(len(x_train)) +
                         'The x_train has' +
                         str(len(y_train)))

    if len(x_test) != len(y_test):
        raise ValueError('shape of x_test and y_test must be equal '
                         'The x_train has ' + str(len(x_test)) +
                         'The y_test has ' +
                         str(len(y_test)))

    np.random.seed(random_state)
    G.setup()
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
