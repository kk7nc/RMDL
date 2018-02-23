import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import numpy as np
from operator import itemgetter
import RMDL_Image
import RMDL_Text
import text_feature_extraction as txt
np.random.seed(7)


def Image_Classifcation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                        n_epochs):

    Data_Type = 1  # 0 if data is text 1 if data is image
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")
        # data_image MINST =1 , CIRFAR = 2, SVHN =3, ORL
        Data_Image = 5
    RMDL_Image.image_classifciation(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                    n_epochs)




def Text_Classifcation(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                        n_epochs):

    Data_Type = 1  # 0 if data is text 1 if data is image
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")

    MAX_SEQUENCE_LENGTH = 500
    MAX_NB_WORDS = 75000
    EMBEDDING_DIM = 100  # it could be 50, 100, and 300 please download GLOVE https://nlp.stanford.edu/projects/glove/ and set address of directory in Text_Data_load.py
    """
    Data_text cloud be 0 to 6 :
    Text 0: WOS5736 that you can download it from:
        http://dx.doi.org/10.17632/9rw3vkcfy4.2
        https://github.com/kk7nc/HDLTex/tree/master/DATA
    Text 1: WOS11967 that you can download it from:
        http://dx.doi.org/10.17632/9rw3vkcfy4.2
        https://github.com/kk7nc/HDLTex/tree/master/DATA
    Text 2: WOS46985 that you can download it from:
        http://dx.doi.org/10.17632/9rw3vkcfy4.2
        https://github.com/kk7nc/HDLTex/tree/master/DATA
    Text 3: reuters 
    Text 4: 20_newsgroup 
    Text 5: IMDB dataset 
    Text 6: manual datasets 
    """
    X_train_tfidf, X_test_tfidf = txt.loadData(X_train, X_test)
    X_train_Glove, X_test_Glove, word_index, embeddings_index = txt.loadData_Tokenizer(X_train, X_test, MAX_NB_WORDS,
                                                                           MAX_SEQUENCE_LENGTH)
    RMDL_Text.Text_classification(X_train_tfidf, X_train_Glove, y_train, X_test_tfidf, X_test_Glove, y_test,
                        Random_Deep,n_epochs,batch_size,sparse_categorical,
                        EMBEDDING_DIM,MAX_SEQUENCE_LENGTH,
                         word_index, embeddings_index)


