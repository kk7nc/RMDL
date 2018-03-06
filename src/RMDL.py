import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import numpy as np
import RMDL_Image
import RMDL_Text
import text_feature_extraction as txt
np.random.seed(7)
import Global
Global.setup()
global GLOVE_DIR
GLOVE_DIR = "D:/glove/" # https://nlp.stanford.edu/projects/glove/

def Image_Classifcation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                        n_epochs):
    RMDL_Image.image_classifciation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                    n_epochs)


def Text_Classifcation(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                        n_epochs):

    MAX_SEQUENCE_LENGTH = 500
    MAX_NB_WORDS = 75000
    EMBEDDING_DIM = 100  # it could be 50, 100, and 300 please download GLOVE https://nlp.stanford.edu/projects/glove/ and set address of directory in Text_Data_load.py

    X_train_tfidf, X_test_tfidf = txt.loadData(X_train, X_test)
    X_train_Glove, X_test_Glove, word_index, embeddings_index = txt.loadData_Tokenizer(X_train, X_test, MAX_NB_WORDS,
                                                                           MAX_SEQUENCE_LENGTH)
    RMDL_Text.Text_classification(X_train_tfidf, X_train_Glove, y_train, X_test_tfidf, X_test_Glove, y_test,
                        Random_Deep,n_epochs,batch_size,sparse_categorical,
                        EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,
                         word_index, embeddings_index)


