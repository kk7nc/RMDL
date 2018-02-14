import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
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
import RMDL_Image
import RMDL_Text
import Plot
import collections
from keras.models import Sequential
import Text_Data_load as Data_load
from sklearn.metrics import f1_score,precision_recall_fscore_support
import BuildModel
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from keras.callbacks import ModelCheckpoint
np.random.seed(7)


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

    MEMORY_MB_MAX = 1600000

    batch_size = 128
    sparse_categorical=0
    n_epochs = [100,100,100] ## DNN--RNN-CNN
    Random_Deep = [0,1,0] ## DNN--RNN-CNN

    Data_Type=0 # 0 if data is text 1 if data is image
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if Data_Type==0:
        MAX_SEQUENCE_LENGTH = 750
        MAX_NB_WORDS = 75000
        EMBEDDING_DIM = 50 #it could be 50, 100, and 300 please download GLOVE https://nlp.stanford.edu/projects/glove/ and set address of directory in Text_Data_load.py
        """
        Data_text cloud be 0 to 6 :
        Text 0: WOS5736 that you can download it from:
            http://dx.doi.org/10.17632/9rw3vkcfy4.2
            https://github.com/kk7nc/HDLTex/tree/master/DATA
        Text 0: WOS5736 that you can download it from:
            http://dx.doi.org/10.17632/9rw3vkcfy4.2
            https://github.com/kk7nc/HDLTex/tree/master/DATA
                Text 0: WOS5736 that you can download it from:
            http://dx.doi.org/10.17632/9rw3vkcfy4.2
            https://github.com/kk7nc/HDLTex/tree/master/DATA
            
        
        """
        Data_text = 0
        RMDL_Text.Text_classification(Data_text,Random_Deep,n_epochs,batch_size,sparse_categorical,EMBEDDING_DIM,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH)
    else:
        #data_image MINST =1 , CIRFAR = 2, or SVHN =3
        Data_Image=3
        RMDL_Image.image_classifciation(Data_Image,Random_Deep,n_epochs,batch_size,sparse_categorical)




