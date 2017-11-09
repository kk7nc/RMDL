import random
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.models import Sequential
import keras
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import numpy as np
from keras.constraints import maxnorm
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D,Conv2D,MaxPooling2D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional,TimeDistributed,Convolution2D,Activation,GlobalAveragePooling2D,Convolution3D,GlobalAveragePooling3D
np.random.seed(7)

def buildModel_DNN_image(shape, nClasses,sparse_categorical):
    model = Sequential()
    values = list(range(128,256))
    Numberof_NOde = random.choice(values)
    Lvalues = list(range(1,5))
    nLayers =random.choice(Lvalues)
    model.add(Flatten(input_shape=shape))
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    for i in range(0,nLayers-1):
        Numberof_NOde = random.choice(values)
        model.add(Dense(Numberof_NOde,activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))


    model.add(Dense(nClasses, activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-2, momentum=0.9, nesterov=True)
    model_tmp = model
    Opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model,model_tmp

def buildModel_DNN_Tex(shape, nClasses,sparse_categorical):
    model = Sequential()
    values = list(range(300,750))
    Numberof_NOde =  random.choice(values)
    values = list(range(1,4))
    nLayers = random.choice(values)
    Numberof_NOde_old = Numberof_NOde
    model.add(Dense(Numberof_NOde,input_dim=shape,activation='relu'))
    model.add(Dropout(0.5))
    values = list(range(300,750))
    for i in range(0,nLayers):
        Numberof_NOde = random.choice(values)
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

def buildModel_RNN_old(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, sparse_categorical):
    model = Sequential()
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
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = (Bidirectional(LSTM(500)))(embedded_sequences)
    # l_lstm = (Bidirectional(LSTM(128)))(l_lstm)
    # l_lstm = (Dense(128))(l_lstm)
    preds = Dense(nClasses, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
    return model

def Image_model_CNN2(num_classes):
    model = Sequential()
    model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32,3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(192, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
# Compile the model

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# define the larger model
def Image_model_CNN(num_classes,shape,kernel_size=(3,3)):
    model = Sequential()
    values = list(range(32,128))
    Layers = list(range(3, 6))
    Layer = 4
    Filter = random.choice(values)
    model.add(Convolution2D(Filter, 3, 3, border_mode='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(Filter, 3, 3))
    model.add(Activation('relu'))
    values = list(range(128, 256))
    for i in range(0,Layer):
        Filter = random.choice(values)
        model.add(Convolution2D(Filter, 3, 3,border_mode='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

   # Filter = random.choice(values)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax',kernel_constraint=maxnorm(3)))

    #model.add(Dense(num_classes,activation="softmax",kernel_constraint=maxnorm(3)))
   # model.add(Dropout(0.2))
    model_tmp = model

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    return model,model_tmp


def Image_3D_model_CNN(num_classes,shape,kernel_size=(3,3)):
    model = Sequential()
    values = list(range(96,256))
    Layer = 3
    Filter = random.choice(values)
    print(shape)
    model.add(Convolution3D(Filter,3,3,1,border_mode='same', input_shape=shape))
    model.add(Activation('relu'))
    model.add(Convolution3D(Filter,3, 3,1, border_mode='same', subsample=(2, 2,1)))
    model.add(Dropout(0.25))
    for i in range(0,Layer):
        Filter = random.choice(values)
        model.add(Convolution3D(Filter, 3,3,1,subsample = (2,2,1)))
        model.add(Activation('relu'))

   # Filter = random.choice(values)
    model.add(Dense(512,activation='relu'))
    model.add(GlobalAveragePooling3D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    #model.add(Dense(num_classes,activation="softmax",kernel_constraint=maxnorm(3)))
   # model.add(Dropout(0.2))


    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    return model

def Image_model_RNN(num_classes,shape):

    values = list(range(90,200))
    values_layer =  list(range(1,5))
    layer = random.choice(values_layer)
    node =  random.choice(values)

    x = Input(shape=shape)

    # Encodes a row of pixels using TimeDistributed Wrapper.
    encoded_rows = TimeDistributed(GRU(node))(x)
    node = random.choice(values)
    # Encodes columns of encoded rows.
    encoded_columns = GRU(node)(encoded_rows)

    # Final predictions and model.
    prediction = Dense(num_classes, activation='softmax')(encoded_columns)
    model = Model(x, prediction)
    model_tmp = model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model,model_tmp



def buildModel_RNN(word_index, embeddings_index, nClasses, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,sparse_categorical):
    model = Sequential()
    values = list(range(90,165))
    values_layer = list(range(1,5))
    layer = random.choice(values_layer)

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
    for i in range(0,layer):
        model.add(GRU(G_L,return_sequences=True, recurrent_dropout=0.2))
        model.add(Dropout(0.25))
    model.add(GRU(G_L, recurrent_dropout=0.2))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nClasses, activation='softmax'))
    model_tmp = model
    if sparse_categorical==0:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return model,model_tmp


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
        values = list(range(1,3))
        filter_sizes = []
        layer = random.choice(values)
        print("Filter  ",layer)
        for fl in range(0,layer):
            filter_sizes.append((fl+3))

        values = list(range(32,100))
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
