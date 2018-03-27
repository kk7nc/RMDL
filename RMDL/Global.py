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
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import numpy as np
import os
import os.path
global GloVe_DIR
from Download import Download_Glove as GloVe
import nltk
nltk.download("all")


MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 75000
EMBEDDING_DIM = 100  # it could be 50, 100, and 300 please download GLOVE https://nlp.stanford.edu/projects/glove/ and set address of directory in Text_Data_load.py

def setup(text=False):
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")

    if text:
        
        global GloVe_DIR
        GloVe_DIR = GloVe.download_and_extract()
        GloVe_file = "glove.6B.100d.txt"
        GloVe_DIR = os.path.join(GloVe_DIR, GloVe_file)
        if not os.path.isfile(GloVe_DIR):
            print("Could not find %s Set GloVe Directory in Global.py ", GloVe)
            exit()
