import numpy as np
import os
global GLOVE_DIR
GLOVE_DIR = "D:/glove/"

def setup():
    np.set_printoptions(threshold=np.inf)
    np.random.seed(7)
    if not os.path.exists(".\weights"):
        os.makedirs(".\weights")
