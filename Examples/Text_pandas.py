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


from RMDL import RMDL_Text as RMDL
import sys
sys.path.append('../Download_datasets')
from sklearn.cross_validation import train_test_split
from RMDL import text_feature_extraction as txt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    file_x = "X.csv"
    file_y = "Y_.csv"
    content = pd.read_csv(file_x, encoding="utf-8")
    Label = pd.read_csv(file_y, encoding="utf-8")
    # content = content.as_matrix()
    content = content.ix[:, 1]
    content = np.array(content).ravel()
    print(np.array(content).transpose().shape)
    Label = Label.as_matrix()
    Label = np.matrix(Label)
    np.random.seed(7)
    # print(Label)
    content = [txt.text_cleaner(x,deep_clean=True) for x in content]
    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.1, random_state=42)
    batch_size = 256
    sparse_categorical = 0
    n_epochs = [100, 100, 100]  ## DNN--RNN-CNN
    Random_Deep = [2, 2, 2]  ## DNN--RNN-CNN

    RMDL.Text_Classification(X_train, y_train, X_test, y_test,
                             batch_size=batch_size,
                             sparse_categorical=True,
                             random_deep=Random_Deep,
                             epochs=n_epochs)