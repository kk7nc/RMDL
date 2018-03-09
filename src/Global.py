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
global GLOVE_DIR
GLOVE_DIR = "D:/glove/"

def setup():
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")
