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

import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import src.RMDL_Text as RMDL

if __name__ == "__main__":
    print("Load Reuters dataset....")

    documents = reuters.fileids()

    train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                documents))
    test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                               documents))
    X_train = [(reuters.raw(doc_id)) for doc_id in train_docs_id]
    X_test = [(reuters.raw(doc_id)) for doc_id in test_docs_id]
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([reuters.categories(doc_id)
                                 for doc_id in train_docs_id])
    y_test = mlb.transform([reuters.categories(doc_id)
                            for doc_id in test_docs_id])
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    batch_size = 100
    sparse_categorical = 0
    n_epochs = [10, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 0, 0]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)
