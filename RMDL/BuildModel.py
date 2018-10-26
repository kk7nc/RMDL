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
from keras.models import Sequential
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten
from keras.layers import Conv1D,MaxPooling2D, \
    MaxPooling1D, Embedding, Dropout,\
    GRU,TimeDistributed,Conv2D,\
    Activation,LSTM,Input
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras import optimizers
import random

def optimizors(random_optimizor):
    if random_optimizor:
        i = random.randint(1,3)
        if i==0:
            opt = optimizers.SGD()
        elif i==1:
            opt= optimizers.RMSprop()
        elif i==2:
            opt= optimizers.Adagrad()
        elif i==3:
            opt = optimizers.Adam()
        elif i==4:
            opt =optimizers.Nadam()
        print(opt)
    else:
        opt= optimizers.Adam()
    return opt



def slice_batch(x, n_gpus, part):
    """
    Divide the input batch into [n_gpus] slices, and obtain slice number [part].
    i.e. if len(x)=10, then slice_batch(x, 2, 1) will return x[5:].
    """

    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def to_multi_gpu(model, n_gpus=2):
    """
    Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.
    """

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:], name="input1")

    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch,
                             lambda shape: shape,
                             arguments={'n_gpus':n_gpus, 'part':g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        merged = Concatenate(axis=0)(towers)

    return Model(inputs=[x], outputs=[merged])


def Build_Model_DNN_Image(shape, number_of_classes, sparse_categorical, min_hidden_layer_dnn,max_hidden_layer_dnn,
                          min_nodes_dnn, max_nodes_dnn, random_optimizor, dropout):
    '''
    buildModel_DNN_image(shape, nClasses,sparse_categorical)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    '''

    model = Sequential()
    values = list(range(min_nodes_dnn,max_nodes_dnn))
    Numberof_NOde = random.choice(values)
    Lvalues = list(range(min_hidden_layer_dnn,max_hidden_layer_dnn))
    nLayers =random.choice(Lvalues)
    print(shape)
    model.add(Flatten(input_shape=shape))
    model.add(Dense(Numberof_NOde,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers-1):
        Numberof_NOde = random.choice(values)
        model.add(Dense(Numberof_NOde,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(number_of_classes, activation='softmax'))
    model_tmp = model
    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    return model,model_tmp


def Build_Model_DNN_Text(shape, nClasses, sparse_categorical,
                         min_hidden_layer_dnn, max_hidden_layer_dnn, min_nodes_dnn,
                         max_nodes_dnn, random_optimizor, dropout):
    """
    buildModel_DNN_Tex(shape, nClasses,sparse_categorical)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    layer = list(range(min_hidden_layer_dnn,max_hidden_layer_dnn))
    node = list(range(min_nodes_dnn, max_nodes_dnn))


    Numberof_NOde =  random.choice(node)
    nLayers = random.choice(layer)

    Numberof_NOde_old = Numberof_NOde
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        Numberof_NOde = random.choice(node)
        model.add(Dense(Numberof_NOde,input_dim=Numberof_NOde_old,activation='relu'))
        model.add(Dropout(dropout))
        Numberof_NOde_old = Numberof_NOde
    model.add(Dense(nClasses, activation='softmax'))
    model_tem = model
    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    return model,model_tem


def Build_Model_CNN_Image(shape, nclasses, sparse_categorical,
                          min_hidden_layer_cnn, max_hidden_layer_cnn, min_nodes_cnn,
                          max_nodes_cnn, random_optimizor, dropout):
    """""
    def Image_model_CNN(num_classes,shape):
    num_classes is number of classes,
    shape is (w,h,p)
    """""

    model = Sequential()
    values = list(range(min_nodes_cnn,max_nodes_cnn))
    Layers = list(range(min_hidden_layer_cnn, max_hidden_layer_cnn))
    Layer = random.choice(Layers)
    Filter = random.choice(values)
    model.add(Conv2D(Filter, (3, 3), padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(Filter, (3, 3)))
    model.add(Activation('relu'))

    for i in range(0,Layer):
        Filter = random.choice(values)
        model.add(Conv2D(Filter, (3, 3),padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nclasses,activation='softmax',kernel_constraint=maxnorm(3)))
    model_tmp = model
    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])

    return model,model_tmp




def Build_Model_RNN_Image(shape,
                          nclasses,
                          sparse_categorical,
                          min_nodes_rnn,
                          max_nodes_rnn,
                          random_optimizor,
                          dropout):
    """
        def Image_model_RNN(num_classes,shape):
        num_classes is number of classes,
        shape is (w,h,p)
    """
    values = list(range(min_nodes_rnn,max_nodes_rnn))
    node =  random.choice(values)

    x = Input(shape=shape)

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(node,recurrent_dropout=dropout))(x)
    node = random.choice(values)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(node,recurrent_dropout=dropout)(encoded_rows)

    # Final predictions and model.
    #prediction = Dense(256, activation='relu')(encoded_columns)
    prediction = Dense(nclasses, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model_tmp = model
    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizors(random_optimizor),
                  metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizors(random_optimizor),
                  metrics=['accuracy'])
    return model,model_tmp


def Build_Model_RNN_Text(word_index, embeddings_index, nclasses,  MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical,
                         min_hidden_layer_rnn, max_hidden_layer_rnn, min_nodes_rnn, max_nodes_rnn, random_optimizor, dropout):
    """
    def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical):
    word_index in word index ,
    embeddings_index is embeddings index, look at data_helper.py
    nClasses is number of classes,
    MAX_SEQUENCE_LENGTH is maximum lenght of text sequences
    """

    model = Sequential()
    values = list(range(min_nodes_rnn,max_nodes_rnn))
    values_layer = list(range(min_hidden_layer_rnn,max_hidden_layer_rnn))

    layer = random.choice(values_layer)
    print(layer)
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) != len(embedding_vector):
                print("could not broadcast input array from shape", str(len(embedding_matrix[i])),
                      "into shape", str(len(embedding_vector)), " Please make sure your"
                                                                " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                exit(1)
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    gru_node = random.choice(values)
    print(gru_node)
    for i in range(0,layer):
        model.add(GRU(gru_node,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
    model.add(GRU(gru_node, recurrent_dropout=0.2))
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))

    model_tmp = model
    #model = to_multi_gpu(model, 3)


    if sparse_categorical:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizors(random_optimizor),
                      metrics=['accuracy'])
    return model,model_tmp


def Build_Model_CNN_Text(word_index, embeddings_index, nclasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical,
                       min_hidden_layer_cnn, max_hidden_layer_cnn, min_nodes_cnn, max_nodes_cnn, random_optimizor,
                       dropout, simple_model=False):

    """
        def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=0):
        word_index in word index ,
        embeddings_index is embeddings index, look at data_helper.py
        nClasses is number of classes,
        MAX_SEQUENCE_LENGTH is maximum lenght of text sequences,
        EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py
        Complexity we have two different CNN model as follows
        F=0 is simple CNN with [1 5] hidden layer
        Complexity=2 is more complex model of CNN with filter_length of range [1 10]
    """

    model = Sequential()
    if simple_model:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) !=len(embedding_vector):
                    print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                     "into shape",str(len(embedding_vector))," Please make sure your"
                                     " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
        values = list(range(min_nodes_cnn,max_nodes_cnn))
        Layer = list(range(min_hidden_layer_cnn,max_hidden_layer_cnn))
        Layer = random.choice(Layer)
        for i in range(0,Layer):
            Filter = random.choice(values)
            model.add(Conv1D(Filter, 5, activation='relu'))
            model.add(Dropout(dropout))
            model.add(MaxPooling1D(5))

        model.add(Flatten())
        Filter = random.choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(dropout))
        Filter = random.choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(dropout))

        model.add(Dense(nclasses, activation='softmax'))
        model_tmp = model
        #model = Model(sequence_input, preds)
        if sparse_categorical:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
    else:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                if len(embedding_matrix[i]) !=len(embedding_vector):
                    print("could not broadcast input array from shape",str(len(embedding_matrix[i])),
                                     "into shape",str(len(embedding_vector))," Please make sure your"
                                     " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)

                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        # applying a more complex convolutional approach
        convs = []
        values_layer = list(range(min_hidden_layer_cnn,max_hidden_layer_cnn))
        filter_sizes = []
        layer = random.choice(values_layer)
        print("Filter  ",layer)
        for fl in range(0,layer):
            filter_sizes.append((fl+2))

        values_node = list(range(min_nodes_cnn,max_nodes_cnn))
        node = random.choice(values_node)
        print("Node  ", node)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            #l_pool = Dropout(0.25)(l_pool)
            convs.append(l_pool)

        l_merge = Concatenate(axis=1)(convs)
        l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
        l_cov1 = Dropout(dropout)(l_cov1)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
        l_cov2 = Dropout(dropout)(l_cov2)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        l_dense = Dense(1024, activation='relu')(l_flat)
        l_dense = Dropout(dropout)(l_dense)
        l_dense = Dense(512, activation='relu')(l_dense)
        l_dense = Dropout(dropout)(l_dense)
        preds = Dense(nclasses, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)
        model_tmp = model
        if sparse_categorical:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizors(random_optimizor),
                          metrics=['accuracy'])


    return model,model_tmp
