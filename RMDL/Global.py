'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification

 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 * Last Update: 04/25/2018
 * This file is part of  RMDL project, University of Virginia.
 * Free to use, change, share and distribute source code of RMDL
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import numpy as np
import os
import os.path
global GloVe_DIR
from RMDL.Download import Download_Glove as GloVe
import nltk
nltk.download("stopwords")
GloVe_file = "glove.6B.50d.txt"
GloVe_directory = ""
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 95000
EMBEDDING_DIM = 50  # it could be 50, 100, and 300 please download GLOVE https://nlp.stanford.edu/projects/glove/ and set address of directory in Text_Data_load.py
global Deep_Model #Deep level
Deep_Model = 1 #Deep_Model = 1 simple 2 deep model 3 very deep model 4 very very deep model
def setup(text=False,GloVe_needed=True):
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")

    if text:
        if GloVe_needed:
            if GloVe_directory=="":
                GloVe_DIR = GloVe.download_and_extract()
            GloVe_DIR = os.path.join(GloVe_DIR, GloVe_file)
            print(GloVe_DIR)
            if not os.path.isfile(GloVe_DIR):
                print("Could not find %s Set GloVe Directory in Global.py ", GloVe)
                exit()
