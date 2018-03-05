import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import text_feature_extraction as txt
from keras.datasets import imdb
import numpy as np
import RMDL


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
    n_epochs = [5000, 500, 1000]  ## DNN--RNN-CNN
    Random_Deep = [1, 30, 0]  ## DNN--RNN-CNN

    RMDL.Text_Classifcation(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)