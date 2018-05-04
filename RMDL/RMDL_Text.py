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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc
import os
import numpy as np
import collections
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.callbacks import ModelCheckpoint
from RMDL import BuildModel as BuildModel
from RMDL.Download import Download_Glove as GloVe
from RMDL import text_feature_extraction as txt
from RMDL import Global as G
from RMDL import Plot as Plot


def Text_Classification(x_train, y_train, x_test,  y_test, batch_size=128,
                        EMBEDDING_DIM=50,MAX_SEQUENCE_LENGTH = 500, MAX_NB_WORDS = 75000,
                        GloVe_dir="", GloVe_file = "glove.6B.50d.txt",
                        sparse_categorical=True, random_deep=[3, 3, 3], epochs=[500, 500, 500],  plot=False,
                        min_hidden_layer_dnn=1, max_hidden_layer_dnn=8, min_nodes_dnn=128, max_nodes_dnn=1024,
                        min_hidden_layer_rnn=1, max_hidden_layer_rnn=5, min_nodes_rnn=32,  max_nodes_rnn=128,
                        min_hidden_layer_cnn=3, max_hidden_layer_cnn=10, min_nodes_cnn=128, max_nodes_cnn=512,
                        random_state=42, random_optimizor=True, dropout=0.5):

    np.random.seed(random_state)


    glove_directory = GloVe_dir
    GloVe_file = GloVe_file



    GloVe_needed = random_deep[1] != 0 or random_deep[2] != 0


    if GloVe_needed:
        if glove_directory == "":
            GloVe_directory = GloVe.download_and_extract()
            GloVe_DIR = os.path.join(GloVe_directory, GloVe_file)
        else:
            GloVe_DIR = os.path.join(glove_directory, GloVe_file)

        if not os.path.isfile(GloVe_DIR):
            print("Could not find %s Set GloVe Directory in Global.py ", GloVe)
            exit()

    G.setup()
    if random_deep[0] != 0:
        x_train_tfidf, x_test_tfidf = txt.loadData(x_train, x_test)
    if random_deep[1] != 0 or random_deep[2] != 0 :
        print(GloVe_DIR)
        x_train_embedded, x_test_embedded, word_index, embeddings_index = txt.loadData_Tokenizer(x_train, x_test,GloVe_DIR,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM)

    del x_train
    del x_test
    gc.collect()

    y_pr = []
    History = []
    score = []
    number_of_classes = np.max(y_train)+1
    i = 0
    while i < random_deep[0]:
        # model_DNN.append(Sequential())
        try:
            print("DNN " + str(i))
            filepath = "weights\weights_DNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            model_DNN, model_tmp = BuildModel.Build_Model_DNN_Text(x_train_tfidf.shape[1],
                                                                   number_of_classes,
                                                                   sparse_categorical,
                                                                   min_hidden_layer_dnn,
                                                                   max_hidden_layer_dnn,
                                                                   min_nodes_dnn,
                                                                   max_nodes_dnn,
                                                                   random_optimizor,
                                                                   dropout)
            model_history = model_DNN.fit(x_train_tfidf, y_train,
                              validation_data=(x_test_tfidf, y_test),
                              epochs=epochs[0],
                              batch_size=batch_size,
                              callbacks=callbacks_list,
                              verbose=2)
            History.append(model_history)

            model_tmp.load_weights(filepath)
            if sparse_categorical:
                model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])

                y_pr_ = model_tmp.predict_classes(x_test_tfidf,
                                                  batch_size=batch_size)
                y_pr.append(np.array(y_pr_))
                score.append(accuracy_score(y_test, y_pr_))
            else:
                model_tmp.compile(loss='categorical_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])

                y_pr_ = model_tmp.predict(x_test_tfidf,
                                          batch_size=batch_size)

                y_pr_ = np.argmax(y_pr_, axis=1)
                y_pr.append(np.array(y_pr_))
                y_test_temp = np.argmax(y_test, axis=1)
                score.append(accuracy_score(y_test_temp, y_pr_))
            # print(y_proba)
            i += 1
            del model_tmp
            del model_DNN
        except:
            print("Error in model", i, "try to re-generate another model")
            if max_hidden_layer_dnn > 3:
                max_hidden_layer_dnn -= 1
            if max_nodes_dnn > 256:
                max_nodes_dnn -= 8

    try:
        del x_train_tfidf
        del x_test_tfidf
        gc.collect()
    except:
        pass

    i=0
    while i < random_deep[1]:
        try:
            print("RNN " + str(i))
            filepath = "weights\weights_RNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath,
                                         monitor='val_acc',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            model_RNN, model_tmp = BuildModel.Build_Model_RNN_Text(word_index,
                                                                   embeddings_index,
                                                                   number_of_classes,
                                                                   MAX_SEQUENCE_LENGTH,
                                                                   EMBEDDING_DIM,
                                                                   sparse_categorical,
                                                                   min_hidden_layer_rnn,
                                                                   max_hidden_layer_rnn,
                                                                   min_nodes_rnn,
                                                                   max_nodes_rnn,
                                                                   random_optimizor,
                                                                   dropout)

            model_history = model_RNN.fit(x_train_embedded, y_train,
                              validation_data=(x_test_embedded, y_test),
                              epochs=epochs[1],
                              batch_size=batch_size,
                              callbacks=callbacks_list,
                              verbose=2)
            History.append(model_history)

            if sparse_categorical:
                model_tmp.load_weights(filepath)
                model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])

                y_pr_ = model_tmp.predict_classes(x_test_embedded, batch_size=batch_size)
                y_pr.append(np.array(y_pr_))
                score.append(accuracy_score(y_test, y_pr_))
            else:
                model_tmp.load_weights(filepath)
                model_tmp.compile(loss='categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                y_pr_ = model_tmp.predict(x_test_embedded, batch_size=batch_size)
                y_pr_ = np.argmax(y_pr_, axis=1)
                y_pr.append(np.array(y_pr_))
                y_test_temp = np.argmax(y_test, axis=1)
                score.append(accuracy_score(y_test_temp, y_pr_))
            i += 1
            del model_tmp
            del model_RNN
            gc.collect()
        except:
            print("Error in model", i, "try to re-generate another model")
            if max_hidden_layer_rnn > 3:
                max_hidden_layer_rnn -= 1
            if max_nodes_rnn > 64:
                max_nodes_rnn -= 2

    gc.collect()

    i = 0
    while i < random_deep[2]:
        try:
            print("CNN " + str(i))

            model_CNN, model_tmp = BuildModel.Build_Model_CNN_Text(word_index,
                                                                   embeddings_index,
                                                                   number_of_classes,
                                                                   MAX_SEQUENCE_LENGTH,
                                                                   EMBEDDING_DIM,
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

            model_history = model_CNN.fit(x_train_embedded, y_train,
                                          validation_data=(x_test_embedded, y_test),
                                          epochs=epochs[2],
                                          batch_size=batch_size,
                                          callbacks=callbacks_list,
                                          verbose=2)
            History.append(model_history)

            model_tmp.load_weights(filepath)
            if sparse_categorical:
                model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
            else:
                model_tmp.compile(loss='categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])

            y_pr_ = model_tmp.predict(x_test_embedded, batch_size=batch_size)
            y_pr_ = np.argmax(y_pr_, axis=1)
            y_pr.append(np.array(y_pr_))

            if sparse_categorical:
                score.append(accuracy_score(y_test, y_pr_))
            else:
                y_test_temp = np.argmax(y_test, axis=1)
                score.append(accuracy_score(y_test_temp, y_pr_))
            i += 1

            del model_tmp
            del model_CNN
            gc.collect()
        except:
            print("Error in model", i, "try to re-generate an other model")
            if max_hidden_layer_cnn > 5:
                max_hidden_layer_cnn -= 1
            if max_nodes_cnn > 128:
                max_nodes_cnn -= 2
                min_nodes_cnn -= 1

    gc.collect()


    y_proba = np.array(y_pr).transpose()

    final_y = []

    for i in range(0, y_proba.shape[0]):
        a = np.array(y_proba[i, :])
        a = collections.Counter(a).most_common()[0][0]
        final_y.append(a)
    if sparse_categorical:
        F_score = accuracy_score(y_test, final_y)
        F1 = precision_recall_fscore_support(y_test, final_y, average='micro')
        F2 = precision_recall_fscore_support(y_test, final_y, average='macro')
        F3 = precision_recall_fscore_support(y_test, final_y, average='weighted')
        cnf_matrix = confusion_matrix(y_test, final_y)
        # Compute confusion matrix
        # Plot non-normalized confusion matrix

        if plot:
            classes = list(range(0, np.max(y_test)+1))
            Plot.plot_confusion_matrix(cnf_matrix, classes=classes,
                                       title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix

            Plot.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                                       title='Normalized confusion matrix')
    else:
        y_test_temp = np.argmax(y_test, axis=1)
        F_score = accuracy_score(y_test_temp, final_y)
        F1 = precision_recall_fscore_support(y_test_temp, final_y, average='micro')
        F2 = precision_recall_fscore_support(y_test_temp, final_y, average='macro')
        F3 = precision_recall_fscore_support(y_test_temp, final_y, average='weighted')
    if plot:
        Plot.RMDL_epoch(History)
    print(y_proba.shape)
    print("Accuracy of",len(score),"models:",score)
    print("Accuracy:",F_score)
    print("F1_Micro:",F1)
    print("F1_Macro:",F2)
    print("F1_weighted:",F3)
