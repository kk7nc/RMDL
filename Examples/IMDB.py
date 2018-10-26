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


from RMDL import text_feature_extraction as txt
from keras.datasets import imdb
import numpy as np
from RMDL import RMDL_Text as RMDL

if __name__ == "__main__":
    print("Load IMDB dataset....")
    MAX_NB_WORDS = 75000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
    print(len(X_train))
    print(y_test)
    word_index = imdb.get_word_index()
    index_word = {v: k for k, v in word_index.items()}
    X_train = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_train]
    X_test = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_test]
    X_train = np.array(X_train)
    X_train = np.array(X_train).ravel()
    print(X_train.shape)
    X_test = np.array(X_test)
    X_test = np.array(X_test).ravel()

    batch_size = 100
    sparse_categorical = 0
    n_epochs = [500, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3,3]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test,
                             batch_size=batch_size,
                             sparse_categorical=sparse_categorical,
                             random_deep=Random_Deep,
                             epochs=n_epochs)
