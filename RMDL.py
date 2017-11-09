import os

from sklearn.metrics import accuracy_score
from keras.datasets import cifar,mnist,imdb
import numpy as np
import itertools
import scipy.io as sio
import matplotlib.pyplot as plt
import gc
from operator import itemgetter
from keras.datasets import cifar10,cifar100
from sklearn.metrics import confusion_matrix
import random
import collections
from keras.models import Sequential
import Data_load
from sklearn.metrics import f1_score,precision_recall_fscore_support
import BuildModel
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from keras.callbacks import ModelCheckpoint
import Plot



np.random.seed(7)

os.environ['THEANO_FLAGS'] = "device=gpu1"

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #if normalize:
      #  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     #   print("Normalized confusion matrix")
    #else:
     #   print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()



def getUniqueWords(allWords) :
    uniqueWords = []
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
def column(matrix,i):
    f = itemgetter(i)
    return map(f,matrix)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])




def FilterByCluster(input_x,input_y,target):
    out = [row for row in input_x if target in input_y[row]]
    return (out)

def keyword_indexing(contentKey):
    vocabulary = list(map(lambda x: x.split(';'), contentKey))
    vocabulary = list(np.unique(list(chain(*vocabulary))))

    vec = CountVectorizer(vocabulary=vocabulary, tokenizer=lambda x: x.split(';'))
    out = np.array(vec.fit_transform(contentKey).toarray())
    print(out.shape)





if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
    #K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32))
    MEMORY_MB_MAX = 1600000
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NB_WORDS = 75000
    EMBEDDING_DIM = 300
    batch_size = 128
    sparse_categorical=0
    n_epochs = [100,100,100] ## DNN--RNN-CNN
    Random_Deep = [4,5,0] ## DNN--RNN-CNN
    MNIST =1
    CIRFAR=1
    text=0
    image=1
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if text==1:
        #fname = "..\DATA\DataSet3\input.txt"
        #fnamek = "..\DATA\DataSet3\l2_2.txt"
        fname = "D:\CHI\X_new.csv"
        fnamek = "D:\CHI\Y_new.csv"
        y_proba = []
        model_DNN = []
        model_RNN = []
        model_CNN = []
        History = []
        score = []
        number_of_classes_L1 =17
        X_train,X_train_M, y_train,X_test, X_test_M, y_test, word_index, embeddings_index = Data_load.Load_data(fname, fnamek,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)

        for i in range(0,Random_Deep[0]):
            #model_DNN.append(Sequential())
            print("DNN "+ str(i))
            filepath = "weights_DNN_" + str(i) + ".hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                         mode='max')
            callbacks_list = [checkpoint]

            model_DNN, model_tmp = BuildModel.buildModel_DNN_Tex(X_train.shape[1],number_of_classes_L1,sparse_categorical)
            h = model_DNN.fit(X_train, y_train,
                         validation_data=(X_test, y_test),
                         epochs=n_epochs[0],
                         batch_size=batch_size,
                         callbacks=callbacks_list,
                         verbose=2)
            History.append(h)

            model_tmp.load_weights("weights_DNN_" + str(i) + ".hdf5")
            if sparse_categorical==0:
                model_tmp.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
                y_proba.append(np.array(y_pr))
                score.append(accuracy_score(y_test, y_pr))
            else:
                model_tmp.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
                y_pr = model_tmp.predict(X_test, batch_size=batch_size)
                y_pr = np.argmax(y_pr,axis=1)
                y_proba.append(np.array(y_pr))
                y_test_temp = np.argmax(y_test,axis=1)
                score.append(accuracy_score(y_test_temp, y_pr))
            #print(y_proba)
            del model_tmp
            del model_DNN
            gc.collect()
            #except:
               #print("Error in model ",i,"   try to re-generate an other model")


        #for i in range(0, Random_Deep[1]):
         #   model_CNN.append(Sequential())
        i=0
        while i <Random_Deep[1]:
            try:
                print("CNN" + str(i))


                model_CNN, model_tmp = BuildModel.buildModel_CNN(word_index, embeddings_index, number_of_classes_L1,
                                                         MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1,sparse_categorical)
                #print(model_CNN[i].summary())

                filepath = "weights_CNN_" + str(i) + ".hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                             mode='max')
                callbacks_list = [checkpoint]


                h =model_CNN.fit(X_train_M, y_train,
                                     validation_data=(X_test_M, y_test),
                                     epochs=n_epochs[1],
                                     batch_size=batch_size,
                                     callbacks=callbacks_list,
                                     verbose=2)
                History.append(h)
                model_tmp.load_weights("weights_CNN_" + str(i) + ".hdf5")
                if sparse_categorical == 0:
                    model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                else:
                    model_tmp.compile(loss='categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                y_pr = model_tmp.predict(X_test_M, batch_size=batch_size)
                y_pr = np.argmax(y_pr, axis=1)
                y_proba.append(np.array(y_pr))

                if sparse_categorical == 0:
                    score.append(accuracy_score(y_test, y_pr))
                else:
                    y_test_temp = np.argmax(y_test,axis=1)
                    score.append(accuracy_score(y_test_temp, y_pr))
                i +=1

                del model_tmp
                del model_CNN
                gc.collect()
            except:
                print("Error in model ", i, "   try to re-generate an other model")


        gc.collect()
        for i in range(0, Random_Deep[2]):
            model_RNN.append(Sequential())
        i = 0
        while i < Random_Deep[2]:
            try:
                print("RNN "  + str(i))
                values = list(range(4))
                Layer = random.choice(values)
                print(int(Layer))
                filepath = "weights_RNN_" + str(i) + ".hdf5"
                checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                             mode='max')
                callbacks_list = [checkpoint]

                model_RNN[i], model_tmp = BuildModel.buildModel_RNN(word_index, embeddings_index, number_of_classes_L1,
                                                         MAX_SEQUENCE_LENGTH, EMBEDDING_DIM,sparse_categorical)
                h = model_RNN[i].fit(X_train_M, y_train,
                                 validation_data=(X_test_M, y_test),
                                 epochs=n_epochs[2],
                                 batch_size=batch_size,
                                 callbacks=callbacks_list,
                                 verbose=2)
                History.append(h)


                if sparse_categorical == 0:
                    model_tmp.load_weights(filepath)
                    model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                    y_pr = model_tmp.predict_classes(X_test_M, batch_size=batch_size)
                    y_proba.append(np.array(y_pr))
                    score.append(accuracy_score(y_test, y_pr))
                else:
                    model_tmp.load_weights(filepath)
                    model_tmp.compile(loss='categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                    y_pr = model_tmp.predict(X_test_M, batch_size=batch_size)
                    y_pr = np.argmax(y_pr,axis=1)
                    y_proba.append(np.array(y_pr))
                    y_test_temp = np.argmax(y_test,axis=1)
                    score.append(accuracy_score(y_test_temp, y_pr))
                i+=1
                del model_tmp
                gc.collect()
            except:
                print("Error in model ", i, "   try to re-generate an other model")
        del model_RNN
        gc.collect()
        Plot.plot_RMDL(History)# plot histori of all RDL models
        y_proba = np.array(y_proba).transpose()
        #print(y_proba.shape)
        final_y = []
        for i in range(0, y_proba.shape[0]):
            a = np.array(y_proba[i, :])
            a = collections.Counter(a).most_common()[0][0]
            final_y.append(a)
        if sparse_categorical == 0:
            F_score = accuracy_score(y_test, final_y)
            F1 = precision_recall_fscore_support(y_test, final_y, average='micro')
            F2 = precision_recall_fscore_support(y_test, final_y, average='macro')
            F3 = precision_recall_fscore_support(y_test, final_y, average='weighted')
            cnf_matrix =  confusion_matrix(y_test, final_y)
            # Compute confusion matrix
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix,classes=[0,1],
                                  title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,
                                  title='Normalized confusion matrix')
        else:
            y_test_temp = np.argmax(y_test, axis=1)
            F_score = accuracy_score(y_test_temp, final_y)
            F1 = precision_recall_fscore_support(y_test_temp, final_y, average='micro')
            F2 = precision_recall_fscore_support(y_test_temp, final_y, average='macro')
            F3 = precision_recall_fscore_support(y_test_temp, final_y, average='weighted')
        print(score)
        print(F_score)
        print(F1)
        print(F2)
        print(F3)



    else:
        if image==1:
            if MNIST ==1:
                num_classes=10;
                y_proba = []
                model = []
                score = []
                history_ = []
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                X_train_D = X_train.reshape(X_train.shape[0], 28, 28).astype('float32')
                X_test_D = X_test.reshape(X_test.shape[0], 28, 28).astype('float32')
                X_train_D = X_train_D / 255.0
                X_test_D = X_test_D / 255.0

                for i in range(0, Random_Deep[0]):
                    print("MNIST DNN ", i, "\n")
                    model.append(Sequential())
                    model[i], model_tmp = BuildModel.buildModel_DNN_image((28, 28), num_classes, 0)
                    filepath = "weights_DNN_" + str(i) + ".hdf5"
                    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
                    callbacks_list = [checkpoint]

                    history = model[i].fit(X_train_D, y_train, validation_data=(X_test_D, y_test),
                                           epochs=n_epochs[0], batch_size=batch_size, callbacks=callbacks_list,
                                           verbose=2)
                    history_.append(history)
                    model_tmp.load_weights("weights_DNN_" + str(i) + ".hdf5")

                    if sparse_categorical == 0:
                        model_tmp.compile(loss='sparse_categorical_crossentropy',
                                          optimizer='adam',
                                          metrics=['accuracy'])
                    else:
                        model_tmp.compile(loss='categorical_crossentropy',
                                          optimizer='adam',
                                          metrics=['accuracy'])

                    y_pr = model_tmp.predict_classes(X_test_D, batch_size=batch_size)
                    y_proba.append(np.array(y_pr))
                    score.append(accuracy_score(y_test, y_pr))


                for i in range(0, Random_Deep[1]):
                    print("RNN ", i,"\n")
                    model.append(Sequential())
                    model[i],model_tmp = BuildModel.Image_model_RNN(num_classes,(28,28))

                    filepath = "weights_RNN_"+str(i)+".hdf5"
                    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
                    callbacks_list = [checkpoint]


                    history = model[i].fit(X_train_D, y_train ,validation_data=(X_test_D, y_test),
                                 epochs=n_epochs[1], batch_size=batch_size,verbose=2,callbacks=callbacks_list)
                    model_tmp.load_weights("weights_RNN_" + str(i) + ".hdf5")
                    model_tmp.compile(loss='sparse_categorical_crossentropy',
                                  optimizer='rmsprop',
                                  metrics=['accuracy'])
                    history_.append(history)


                    y_pr = model_tmp.predict(X_test_D, batch_size=batch_size)
                    y_pr = np.argmax(y_pr,axis=1)
                    y_proba.append(np.array(y_pr))
                    score.append(accuracy_score(y_test, y_pr))




                # reshape to be [samples][pixels][width][height]

                print(np.array(X_train).shape)
                X_train = X_train.reshape(X_train.shape[0], 28, 28,1).astype('float32')
                X_test = X_test.reshape(X_test.shape[0], 28, 28,1).astype('float32')
                X_train = X_train / 255.0
                X_test = X_test / 255.0

                for i in range(0, Random_Deep[2]):
                    print("CNN ",i,"\n")
                    model.append(Sequential())
                    model[i], model_tmp = BuildModel.Image_model_CNN(num_classes,(28,28,1))


                    filepath = "weights_CNN_"+str(i)+".hdf5"
                    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
                    callbacks_list = [checkpoint]


                    history = model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                                 epochs=n_epochs[2], batch_size=batch_size,callbacks=callbacks_list,verbose=2)
                    history_.append(history)
                    model_tmp.load_weights("weights_CNN_" + str(i) + ".hdf5")
                    model_tmp.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                                  metrics=['accuracy'])
                    y_pr = model_tmp.predict_classes(X_test, batch_size=batch_size)
                    y_proba.append(np.array(y_pr))
                    score.append(accuracy_score(y_test, y_pr))

                iter = Random_Deep[0]+Random_Deep[1]+Random_Deep[2]

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
                plot_confusion_matrix(cnf_matrix, classes=[0, 1,2,3,4,5,6,7,8,9],
                                      title='Confusion matrix, without normalization')

                # Plot normalized confusion matrix
                plt.figure()
                plot_confusion_matrix(cnf_matrix, classes=[0, 1,2,3,4,5,6,7,8,9], normalize=True,
                                      title='Normalized confusion matrix')
                print(score)
                print(F_score)
                print(F1)
                print(F2)
                print(F3)
            else:
                if CIRFAR==1:
                    (X_train, y_train), (X_test, y_test) = cifar10.load_data()  # fetch CIFAR-100 data
                    plt.imshow(X_train[120])
                    plt.show()
                    X_train_D = X_train.astype('float32')
                    X_test_D = X_test.astype('float32')
                    X_train_D = X_train_D / 255.0
                    X_test_D = X_test_D / 255.0
                    print(X_train_D.shape)
                    print(type(X_train))
                    num_classes = np.unique(y_train).shape[0]  # there are 100 image classes


                    print(num_classes)

                    print(X_train.shape)

                    y_proba = []
                    model = []
                    score = []
                    history_ = []
                    for i in range(0, Random_Deep[0]):
                        print("MNIST DNN ", i, "\n")
                        model.append(Sequential())
                        model[i], model_tmp = BuildModel.buildModel_DNN_image((32, 32,3), num_classes, 0)
                        filepath = "weights_DNN_" + str(i) + ".hdf5"
                        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True,
                                                     mode='max')
                        callbacks_list = [checkpoint]

                        history = model[i].fit(X_train_D, y_train, validation_data=(X_test_D, y_test),
                                               epochs=n_epochs[0], batch_size=batch_size, callbacks=callbacks_list,
                                               verbose=2)
                        history_.append(history)
                        model_tmp.load_weights("weights_DNN_" + str(i) + ".hdf5")

                        if sparse_categorical == 0:
                            model_tmp.compile(loss='sparse_categorical_crossentropy',
                                              optimizer='adam',
                                              metrics=['accuracy'])
                        else:
                            model_tmp.compile(loss='categorical_crossentropy',
                                              optimizer='adam',
                                              metrics=['accuracy'])

                        y_pr = model_tmp.predict_classes(X_test_D, batch_size=batch_size)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))

                    for i in range(0, Random_Deep[2]):
                        print("RNN ", i, "\n")
                        model.append(Sequential())
                        model[i], model_tmp = BuildModel.Image_model_RNN(num_classes, (32, 32,3))

                        filepath = "weights_RNN_" + str(i) + ".hdf5"
                        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                     mode='max')
                        callbacks_list = [checkpoint]

                        history = model[i].fit(X_train_D, y_train, validation_data=(X_test_D, y_test),
                                               epochs=n_epochs[2], batch_size=batch_size, verbose=1,
                                               callbacks=callbacks_list)
                        model_tmp.load_weights("weights_RNN_" + str(i) + ".hdf5")
                        model_tmp.compile(loss='sparse_categorical_crossentropy',
                                          optimizer='rmsprop',
                                          metrics=['accuracy'])
                        history_.append(history)

                        y_pr = model_tmp.predict(X_test_D, batch_size=batch_size)
                        y_pr = np.argmax(y_pr, axis=1)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))

                    # reshape to be [samples][pixels][width][height]


                    for i in range(0, Random_Deep[1]):
                        print("CNN ", i, "\n")
                        model.append(Sequential())
                        model[i], model_tmp = BuildModel.Image_model_CNN(num_classes, (32, 32,3))

                        filepath = "weights_CNN_" + str(i) + ".hdf5"
                        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                     mode='max')
                        callbacks_list = [checkpoint]

                        history = model[i].fit(X_train_D, y_train, validation_data=(X_test_D, y_test),
                                               epochs=n_epochs[1], batch_size=batch_size, callbacks=callbacks_list,
                                               verbose=2)
                        history_.append(history)
                        model_tmp.load_weights("weights_CNN_" + str(i) + ".hdf5")
                        model_tmp.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                                          metrics=['accuracy'])
                        y_pr = model_tmp.predict_classes(X_test_D, batch_size=batch_size)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))

                    iter = Random_Deep[0] + Random_Deep[1] + Random_Deep[2]
                    plt.legend(
                        ['RDL 1', 'RDL 2', 'RDL 3', 'RDL 4', 'RDL 5', 'RDL 6', 'RDL 7', 'RDL 8', 'RDL 9', 'RDL 10',
                         'RDL 11', 'RDL 12', 'RDL 13', 'RDL 14', 'RDL 15'], loc='upper right')
                    for i in range(0, iter):
                        plt.plot(history_[i].history['acc'])
                        plt.title('model accuracy')
                        plt.ylabel('accuracy')
                        plt.xlabel('epoch')
                    plt.legend(
                        ['RDL 1', 'RDL 2', 'RDL 3', 'RDL 4', 'RDL 5', 'RDL 6', 'RDL 7', 'RDL 8', 'RDL 9', 'RDL 10',
                         'RDL 11', 'RDL 12', 'RDL 13', 'RDL 14', 'RDL 15'], loc='upper right')
                    for i in range(0, iter):
                        plt.plot(history_[i].history['val_acc'])
                        plt.title('model accuracy')
                        plt.ylabel('accuracy')
                        plt.xlabel('epoch')

                    plt.show()
                    for i in range(0, iter):
                        # summarize history for loss
                        plt.plot(history_[i].history['loss'])

                        plt.title('model loss train')
                        plt.ylabel('loss')
                        plt.xlabel('epoch')

                    plt.legend(
                        ['RDL 1', 'RDL 2', 'RDL 3', 'RDL 4', 'RDL 5', 'RDL 6', 'RDL 7', 'RDL 8', 'RDL 9', 'RDL 10',
                         'RDL 11', 'RDL 12', 'RDL 13', 'RDL 14', 'RDL 15'], loc='upper right')
                    plt.show()
                    for i in range(0, iter):
                        # summarize history for loss
                        plt.plot(history_[i].history['val_loss'])

                        plt.title('model loss test')
                        plt.ylabel('loss')
                        plt.xlabel('epoch')

                    plt.legend(
                        ['RDL 1', 'RDL 2', 'RDL 3', 'RDL 4', 'RDL 5', 'RDL 6', 'RDL 7', 'RDL 8', 'RDL 9', 'RDL 10',
                         'RDL 11', 'RDL 12', 'RDL 13', 'RDL 14', 'RDL 15'], loc='upper right')
                    plt.show()

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
                    plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                          title='Confusion matrix, without normalization')

                    # Plot normalized confusion matrix
                    plt.figure()
                    plot_confusion_matrix(cnf_matrix, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], normalize=True,
                                          title='Normalized confusion matrix')
                    print(score)
                    print(F_score)
                    print(F1)
                    print(F2)
                    print(F3)
                else:

                    train_data = sio.loadmat("D:\\Google\\RDeeps\\DATA\image\\train_32x32.mat")
                    test_data = sio.loadmat("D:\\Google\\RDeeps\DATA\\image\\test_32x32.mat")

                    # access to the dict
                    X_tr = train_data['X']
                    y_train = train_data['y']

                    X_te = test_data['X']
                    y_test = test_data['y']




                    x_train = []

                    for i in range(X_tr.shape[3]):
                        x_train.append(X_tr[:,:,:,i])

                    x_test = []
                    LLL = np.array(X_te[:, :, :, 234])
                    print(LLL.shape)
                    for i in range(X_te.shape[3]):
                        x_test.append(X_te[:,:,:,i])





                   # print(y_train)

                    plt.imshow(x_train[12])
                   # plt.show()

                    X_train = np.array(x_train)
                    X_test = np.array(x_test)

                    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3).astype('float32')
                    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')

                    X_train = X_train / 256
                    X_test = X_test / 256
                    y_train -= 1
                    y_test -= 1
                    #print(X_train)
                    print(np.max(y_train), np.min(y_train))
                    num_classes = np.unique(y_train).shape[0]

                    print("number of classes:  "+str(num_classes))

                    y_proba = []
                    model = []
                    score = []
                    for i in range(0, Random_Deep[0]):
                        print("DNN ", i, "\n")
                        model.append(Sequential())
                        model[i] = BuildModel.buildModel_DNN_image((32, 32 , 3), num_classes, 0)
                        model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                                     epochs=n_epochs[0], batch_size=batch_size,verbose=2)
                        y_pr = model[i].predict_classes(X_test, batch_size=batch_size)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))




                    for i in range(0, Random_Deep[2]):
                        print("CNN ", i, "\n")
                        model.append(Sequential())
                        model[i] = BuildModel.Image_model_CNN(num_classes, (32, 32, 3))
                        model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                                     epochs=n_epochs[1], batch_size=batch_size,verbose=2)
                        y_pr = model[i].predict(X_test, batch_size=batch_size)
                        y_pr = np.argmax(y_pr, axis=1)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))

                    for i in range(0, Random_Deep[1]):
                        print("RNN ", i, "\n")
                        model.append(Sequential())
                        model[i] = BuildModel.Image_model_RNN(num_classes, (32, 32, 3))
                        model[i].fit(X_train, y_train, validation_data=(X_test, y_test),
                                     epochs=n_epochs[2], batch_size=batch_size,verbose=2)
                        y_pr = model[i].predict(X_test, batch_size=batch_size)
                        y_pr = np.argmax(y_pr, axis=1)
                        y_proba.append(np.array(y_pr))
                        score.append(accuracy_score(y_test, y_pr))
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
                    precision = precision_recall_fscore_support(y_test, final_y, average='micro')
                    # recall = recall_score(y_test, final_y,average='micro')
                    print(score)
                    print(F_score)
                    print(F1)
                    print(F2)
                    print(F3)
                    print(precision)
                    # print(recall)




