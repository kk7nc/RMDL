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

import random
from keras.models import Sequential
import keras
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Flatten
from keras.layers import Conv1D,MaxPooling2D, \
    MaxPooling1D, Embedding, Merge, Dropout,\
    GRU,TimeDistributed,Conv2D,\
    Activation,Conv3D,GlobalAveragePooling3D,LSTM
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate
import tensorflow as tf


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

'''
buildModel_DNN_image(shape, nClasses,sparse_categorical)
Build Deep neural networks Model for text classification
Shape is input feature space
nClasses is number of classes
'''

def buildModel_DNN_image(shape, nClasses,sparse_categorical):
    model = Sequential()
    values = list(range(128,1024))
    Numberof_NOde = random.choice(values)
    Lvalues = list(range(1,5))
    nLayers =random.choice(Lvalues)
    print(shape)
    model.add(Flatten(input_shape=shape))
    model.add(Dense(Numberof_NOde,activation='relu'))
    model.add(Dropout(0.25))
    for i in range(0,nLayers-1):
        Numberof_NOde = random.choice(values)
        model.add(Dense(Numberof_NOde,activation='relu'))
        model.add(Dropout(0.25))
    model.add(Dense(nClasses, activation='softmax'))
    model_tmp = model
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model,model_tmp


'''
buildModel_DNN_Tex(shape, nClasses,sparse_categorical)
Build Deep neural networks Model for text classification
Shape is input feature space
nClasses is number of classes
'''

def buildModel_DNN_Tex(shape, nClasses,sparse_categorical):
    model = Sequential()
    node = list(range(256,1024))
    Numberof_NOde =  random.choice(node)
    layer = list(range(1,4))
    nLayers = random.choice(layer)
    Numberof_NOde_old = Numberof_NOde
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(Dropout(0.5))
    for i in range(0,nLayers):
        Numberof_NOde = random.choice(node)
        model.add(Dense(Numberof_NOde,input_dim=Numberof_NOde_old,activation='relu'))
        model.add(Dropout(0.5))
        Numberof_NOde_old = Numberof_NOde
    model.add(Dense(nClasses, activation='softmax'))
    model_tem = model
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model,model_tem

'''
def Image_model_CNN(num_classes,shape):
num_classes is number of classes, 
shape is (w,h,p) 
'''

def Image_model_CNN(num_classes,shape):
    model = Sequential()
    values = list(range(32,256))
    Layers = list(range(1, 4))
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
        model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax',kernel_constraint=maxnorm(3)))
    model_tmp = model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    return model,model_tmp

'''
def Image_3D_model_CNN(num_classes,shape):
num_classes is number of classes, 
shape is (w,h,p) 
'''
def Image_3D_model_CNN(num_classes,shape,kernel_size=(3,3)):
    model = Sequential()
    values = list(range(96,256))
    Layer = 3
    Filter = random.choice(values)
    print(shape)
    model.add(Conv2D(Filter,(3,3),1,padding='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Conv2D(Filter,(3, 3),1, padding='same', subsample=(2, 2,1)))
    model.add(Dropout(0.25))
    for i in range(0,Layer):
        Filter = random.choice(values)
        model.add(Conv3D(Filter,(3,3,1),subsample = (2,2,1)))
        model.add(Activation('relu'))
    model.add(Dense(512,activation='relu'))
    model.add(GlobalAveragePooling3D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    return model
'''
def Image_model_RNN(num_classes,shape):
num_classes is number of classes, 
shape is (w,h,p) 
'''
def Image_model_RNN(num_classes,shape):

    values = list(range(128,512))
    node =  random.choice(values)

    x = Input(shape=shape)

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(LSTM(node))(x)
    node = random.choice(values)
    # Encodes columns of encoded rows.
    encoded_columns = LSTM(node)(encoded_rows)
    #encoded_columns = GRU(node)(encoded_columns)

    # Final predictions and model.
    #prediction = Dense(256, activation='relu')(encoded_columns)
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model_tmp = model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model,model_tmp


'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences
'''
def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,sparse_categorical):
    model = Sequential()
    values = list(range(32,128))
    values_layer = list(range(1,5))

    layer = random.choice(values_layer)
    print(layer)
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True))

    G_L = random.choice(values)
    print(G_L)
    for i in range(0,layer):
        model.add(GRU(G_L,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(0.25))
    model.add(GRU(G_L, recurrent_dropout=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))

    model_tmp = model
    #model = to_multi_gpu(model, 3)


    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model,model_tmp

'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,Complexity=0):
word_index in word index , 
embeddings_index is embeddings index, look at data_helper.py 
nClasses is number of classes, 
MAX_SEQUENCE_LENGTH is maximum lenght of text sequences, 
EMBEDDING_DIM is an int value for dimention of word embedding look at data_helper.py 
Complexity we have two different CNN model as follows 
F=0 is simple CNN with [1 5] hidden layer
Complexity=2 is more complex model of CNN with filter_length of range [1 10]
'''
def buildModel_CNN(word_index,embeddings_index,nClasses,MAX_SEQUENCE_LENGTH,EMBEDDING_DIM,F=1,sparse_categorical=0):
    model = Sequential()
    if F==0:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True))
       # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
        #embedded_sequences = embedding_layer(sequence_input)
        values = [256]
        Layer = list(range(1,5))
        Layer = random.choice(Layer)
        for i in range(0,Layer):
            Filter = random.choice(values)
            model.add(Conv1D(Filter, 5, activation='relu'))
            model.add(Dropout(0.2))
            model.add(MaxPooling1D(5))

        model.add(Flatten())
        Filter = random.choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(0.2))
        Filter = random.choice(values)
        model.add(Dense(Filter, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(nClasses, activation='softmax'))
        model_tmp = model
        #model = Model(sequence_input, preds)
        if sparse_categorical == 0:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
    else:
        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(len(word_index) + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)

        # applying a more complex convolutional approach
        convs = []
        values = list(range(3,10))
        filter_sizes = []
        layer = random.choice(values)
        print("Filter  ",layer)
        for fl in range(0,layer):
            filter_sizes.append((fl+2))

        values = list(range(128,400))
        node = random.choice(values)
        print("Node  ", node)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(node, kernel_size=fsz, activation='relu')(embedded_sequences)
            l_pool = MaxPooling1D(5)(l_conv)
            #l_pool = Dropout(0.25)(l_pool)
            convs.append(l_pool)

        l_merge = Merge(mode='concat', concat_axis=1)(convs)
        l_cov1 = Conv1D(node, 5, activation='relu')(l_merge)
        l_cov1 = Dropout(0.25)(l_cov1)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        l_cov2 = Conv1D(node, 5, activation='relu')(l_pool1)
        l_cov2 = Dropout(0.25)(l_cov2)
        l_pool2 = MaxPooling1D(30)(l_cov2)
        l_flat = Flatten()(l_pool2)
        values = list(range(250,1000))
        node = random.choice(values)
        l_dense = Dense(node, activation='relu')(l_flat)
        l_dense = Dropout(0.5)(l_dense)
        preds = Dense(nClasses, activation='softmax')(l_dense)
        model = Model(sequence_input, preds)
        model_tmp = model
        if sparse_categorical == 0:
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])


    return model,model_tmp
