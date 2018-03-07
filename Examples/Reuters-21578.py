import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import text_feature_extraction as txt
from nltk.corpus import reuters
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import RMDL

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
    n_epochs = [5000, 500, 500]  ## DNN--RNN-CNN
    Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN

    RMDL.Text_Classifcation(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)