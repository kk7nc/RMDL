import sys
sys.path.append('../src')
sys.path.append('../Download_datasets')
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="2,1,0"
import text_feature_extraction as txt
from sklearn.model_selection import train_test_split
import WOS_input as WOS
import numpy as np
import RMDL

if __name__ == "__main__":
    WOS.download_and_extract()
    fname = "./Data_WOS/WebOfScience/WOS46985/X.txt"
    fnamek = "./Data_WOS/WebOfScience/WOS46985/Y.txt"
    with open(fname, encoding="utf-8") as f:
        content = f.readlines()
        content = [txt.text_cleaner(x) for x in content]
    with open(fnamek) as fk:
        contentk = fk.readlines()
    contentk = [x.strip() for x in contentk]
    Label = np.matrix(contentk, dtype=int)
    Label = np.transpose(Label)

    np.random.seed(7)
    print(Label.shape)
    X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=4)

    batch_size = 100
    sparse_categorical = 0
    n_epochs = [5000, 500, 1000]  ## DNN--RNN-CNN
    Random_Deep = [1, 30, 0]  ## DNN--RNN-CNN

    RMDL.Text_Classifcation(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                            n_epochs)