[![DOI](https://img.shields.io/badge/DOI-10.1145/3206098.3206111-blue.svg?style=flat)](https://doi.org/10.1145/3206098.3206111)
[![Pypi](https://img.shields.io/badge/Pypi-RMDL%201.0.0-blue.svg)](https://pypi.org/project/RMDL/)
[![werckerstatus](https://app.wercker.com/status/3a564158e809971e2f7416beba5d05af/s/master)](https://app.wercker.com/project/byKey/3a564158e809971e2f7416beba5d05af)
[![appveyor](https://ci.appveyor.com/api/projects/status/github/kk7nc/RMDL?branch=master&svg=true)](https://ci.appveyor.com/project/kk7nc/RMDL)
[![BuildStatus](https://travis-ci.org/kk7nc/RMDL.svg?branch=master)](https://travis-ci.org/kk7nc/RMDL)
[![PowerPoint](https://img.shields.io/badge/Presentation-download-red.svg?style=flat)](https://github.com/kk7nc/RMDL/blob/master/Documents/RMDL.pdf)
[![researchgate](https://img.shields.io/badge/ResearchGate-RMDL-blue.svg?style=flat)](https://www.researchgate.net/publication/324922651_RMDL_Random_Multimodel_Deep_Learning_for_Classification)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/kk7nc/RMDL/master)
[![pdf](https://img.shields.io/badge/pdf-download-red.svg?style=flat)](https://github.com/kk7nc/RMDL/blob/master/Documents/ACM-RMDL.pdf)
[![GitHublicense](https://img.shields.io/badge/licence-GPL-blue.svg)](./LICENSE)

Referenced paper : [RMDL: Random Multimodel Deep Learning for
Classification](https://www.researchgate.net/publication/324922651_RMDL_Random_Multimodel_Deep_Learning_for_Classification)


# RMDL: Random Multimodel Deep Learning for Classification #

## Global.py ##

Create weights folder and download GloVe for text classification (if you already download GloVe set Glove Directory in Global.py)


## Text Feature Extraction: ##

We used two different feature extraction : 

### [Term  Frequency-Inverse  Document  Frequency](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) ##

### [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) ##

## BuildModel.py: ##

This file contain build random model of deep learning architectures for image and text including (DNN, CNN, and RNN)



## General requirements: ##

- Python 3.5 or later see [Instruction Documents](https://www.python.org/)

- TensorFlow see [Instruction Documents](https://www.tensorflow.org/install/install_linux).

- scikit-learn see [Instruction Documents](http://scikit-learn.org/stable/install.html)

- Keras see [Instruction Documents](https://keras.io/)

- scipy see [Instruction Documents](https://www.scipy.org/install.html)

- GPU (if you want to run on GPU):

    * CUDA® Toolkit 8.0. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cuda-toolkit). 

    * The [NVIDIA drivers associated with CUDA Toolkit 8.0](http://www.nvidia.com/Download/index.aspx).

    * cuDNN v6. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cudnn). 

    * GPU card with CUDA Compute Capability 3.0 or higher.

    * The libcupti-dev library,

               
Parameters:
===========

Text Classification
--------------------

```python
from RMDL import RMDL_Text
```

```python
Text_Classification(x_train, y_train, x_test,  y_test, batch_size=128,
                 EMBEDDING_DIM=50,MAX_SEQUENCE_LENGTH = 500,
                 MAX_NB_WORDS = 75000, GloVe_dir="",
                 GloVe_file = "glove.6B.50d.txt",
                 sparse_categorical=True, random_deep=[3, 3, 3],
                 epochs=[500, 500, 500],  plot=True,
                 min_hidden_layer_dnn=1, max_hidden_layer_dnn=8,
                 min_nodes_dnn=128, max_nodes_dnn=1024,
                 min_hidden_layer_rnn=1, max_hidden_layer_rnn=5,
                 min_nodes_rnn=32,  max_nodes_rnn=128,
                 min_hidden_layer_cnn=3, max_hidden_layer_cnn=10,
                 min_nodes_cnn=128, max_nodes_cnn=512,
                 random_state=42, random_optimizor=True, dropout=0.05):
```

### Input

-   x\_train
-   y\_train
-   x\_test
-   y\_test

### batch\_size

-   batch\_size: Integer. Number of samples per gradient update. If
    unspecified, it will default to 128.

### EMBEDDING\_DIM

-   batch\_size: Integer. Shape of word embedding (this number should be
    same with GloVe or other pre-trained embedding techniques that be
    used), it will default to 50 that used with pain of glove.6B.50d.txt
    file.

### MAX\_SEQUENCE\_LENGTH

-   MAX\_SEQUENCE\_LENGTH: Integer. Maximum length of sequence or
    document in datasets, it will default to 500.

### MAX\_NB\_WORDS

-   MAX\_NB\_WORDS: Integer. Maximum number of unique words in datasets,
    it will default to 75000.

### GloVe\_dir

-   GloVe\_dir: String. Address of GloVe or any pre-trained directory,
    it will default to null which glove.6B.zip will be download.

### GloVe\_file

-   GloVe\_dir: String. Which version of GloVe or pre-trained word
    emending will be used, it will default to glove.6B.50d.txt.
-   NOTE: if you use other version of GloVe EMBEDDING\_DIM must be same
    dimensions.

### sparse\_categorical

-   sparse\_categorical: bool. When target\'s dataset is (n,1) should be
    True, it will default to True.

### random\_deep

-   random\_deep: Integer \[3\]. Number of ensembled model used in RMDL
    random\_deep\[0\] is number of DNN, random\_deep\[1\] is number of
    RNN, random\_deep\[0\] is number of CNN, it will default to \[3, 3,
    3\].

### epochs

-   epochs: Integer \[3\]. Number of epochs in each ensembled model used
    in RMDL epochs\[0\] is number of epochs used in DNN, epochs\[1\] is
    number of epochs used in RNN, epochs\[0\] is number of epochs used
    in CNN, it will default to \[500, 500, 500\].

### plot

-   plot: bool. True: shows confusion matrix and accuracy and loss

### min\_hidden\_layer\_dnn

-   min\_hidden\_layer\_dnn: Integer. Lower Bounds of hidden layers of
    DNN used in RMDL, it will default to 1.

### max\_hidden\_layer\_dnn

-   max\_hidden\_layer\_dnn: Integer. Upper bounds of hidden layers of
    DNN used in RMDL, it will default to 8.

### min\_nodes\_dnn

-   min\_nodes\_dnn: Integer. Lower bounds of nodes in each layer of DNN
    used in RMDL, it will default to 128.

### max\_nodes\_dnn

-   max\_nodes\_dnn: Integer. Upper bounds of nodes in each layer of DNN
    used in RMDL, it will default to 1024.

### min\_hidden\_layer\_rnn

-   min\_hidden\_layer\_rnn: Integer. Lower Bounds of hidden layers of
    RNN used in RMDL, it will default to 1.

### max\_hidden\_layer\_rnn

-   man\_hidden\_layer\_rnn: Integer. Upper Bounds of hidden layers of
    RNN used in RMDL, it will default to 5.

### min\_nodes\_rnn

-   min\_nodes\_rnn: Integer. Lower bounds of nodes (LSTM or GRU) in
    each layer of RNN used in RMDL, it will default to 32.

### max\_nodes\_rnn

-   max\_nodes\_rnn: Integer. Upper bounds of nodes (LSTM or GRU) in
    each layer of RNN used in RMDL, it will default to 128.

### min\_hidden\_layer\_cnn

-   min\_hidden\_layer\_cnn: Integer. Lower Bounds of hidden layers of
    CNN used in RMDL, it will default to 3.

### max\_hidden\_layer\_cnn

-   max\_hidden\_layer\_cnn: Integer. Upper Bounds of hidden layers of
    CNN used in RMDL, it will default to 10.

### min\_nodes\_cnn

-   min\_nodes\_cnn: Integer. Lower bounds of nodes (2D convolution
    layer) in each layer of CNN used in RMDL, it will default to 128.

### max\_nodes\_cnn

-   min\_nodes\_cnn: Integer. Upper bounds of nodes (2D convolution
    layer) in each layer of CNN used in RMDL, it will default to 512.

### random\_state

-   random\_state : Integer, RandomState instance or None, optional
    (default=None)

    -   If Integer, random\_state is the seed used by the random number generator;

### random\_optimizor

-   random\_optimizor : bool, If False, all models use adam optimizer.
    If True, all models use random optimizers. it will default to True

### dropout

-   dropout: Float between 0 and 1. Fraction of the units to drop for
    the linear transformation of the inputs.

Image Classification
---------------------

```python
from RMDL import RMDL_Image
```

```python
Image_Classification(x_train, y_train, x_test, y_test, shape, batch_size=128,
                     sparse_categorical=True, random_deep=[3, 3, 3],
                     epochs=[500, 500, 500], plot=True,
                     min_hidden_layer_dnn=1, max_hidden_layer_dnn=8,
                     min_nodes_dnn=128, max_nodes_dnn=1024,
                     min_hidden_layer_rnn=1, max_hidden_layer_rnn=5,
                     min_nodes_rnn=32, max_nodes_rnn=128,
                     min_hidden_layer_cnn=3, max_hidden_layer_cnn=10,
                     min_nodes_cnn=128, max_nodes_cnn=512,
                     random_state=42, random_optimizor=True, dropout=0.05)
```

### Input

-   x\_train
-   y\_train
-   x\_test
-   y\_test

### shape

-   shape: np.shape . shape of image. The most common situation would be
    a 2D input with shape (batch\_size, input\_dim).

### batch\_size

-   batch\_size: Integer. Number of samples per gradient update. If
    unspecified, it will default to 128.

### sparse\_categorical

-   sparse\_categorical: bool. When target\'s dataset is (n,1) should be
    True, it will default to True.

### random\_deep

-   random\_deep: Integer \[3\]. Number of ensembled model used in RMDL
    random\_deep\[0\] is number of DNN, random\_deep\[1\] is number of
    RNN, random\_deep\[0\] is number of CNN, it will default to \[3, 3,
    3\].

### epochs

-   epochs: Integer \[3\]. Number of epochs in each ensembled model used
    in RMDL epochs\[0\] is number of epochs used in DNN, epochs\[1\] is
    number of epochs used in RNN, epochs\[0\] is number of epochs used
    in CNN, it will default to \[500, 500, 500\].

### plot

-   plot: bool. True: shows confusion matrix and accuracy and loss

### min\_hidden\_layer\_dnn

-   min\_hidden\_layer\_dnn: Integer. Lower Bounds of hidden layers of
    DNN used in RMDL, it will default to 1.

### max\_hidden\_layer\_dnn

-   max\_hidden\_layer\_dnn: Integer. Upper bounds of hidden layers of
    DNN used in RMDL, it will default to 8.

### min\_nodes\_dnn

-   min\_nodes\_dnn: Integer. Lower bounds of nodes in each layer of DNN
    used in RMDL, it will default to 128.

### max\_nodes\_dnn

-   max\_nodes\_dnn: Integer. Upper bounds of nodes in each layer of DNN
    used in RMDL, it will default to 1024.

### min\_nodes\_rnn

-   min\_nodes\_rnn: Integer. Lower bounds of nodes (LSTM or GRU) in
    each layer of RNN used in RMDL, it will default to 32.

### max\_nodes\_rnn

-   maz\_nodes\_rnn: Integer. Upper bounds of nodes (LSTM or GRU) in
    each layer of RNN used in RMDL, it will default to 128.

### min\_hidden\_layer\_cnn

-   min\_hidden\_layer\_cnn: Integer. Lower Bounds of hidden layers of
    CNN used in RMDL, it will default to 3.

### max\_hidden\_layer\_cnn

-   max\_hidden\_layer\_cnn: Integer. Upper Bounds of hidden layers of
    CNN used in RMDL, it will default to 10.

### min\_nodes\_cnn

-   min\_nodes\_cnn: Integer. Lower bounds of nodes (2D convolution
    layer) in each layer of CNN used in RMDL, it will default to 128.

### max\_nodes\_cnn

-   min\_nodes\_cnn: Integer. Upper bounds of nodes (2D convolution
    layer) in each layer of CNN used in RMDL, it will default to 512.

### random\_state

-   random\_state : Integer, RandomState instance or None, optional
    (default=None)

    -  If Integer, random\_state is the seed used by the random number generator;

### random\_optimizor

-   random\_optimizor : bool, If False, all models use adam optimizer.
    If True, all models use random optimizers. it will default to True

### dropout

-   dropout: Float between 0 and 1. Fraction of the units to drop for
    the linear transformation of the inputs.


## Error and Comments: ##

Send an email to [kk7nc@virginia.edu](mailto:kk7nc@virginia.edu)


## Citation ##

    @inproceedings{Kowsari2018RMDL,
    title={RMDL: Random Multimodel Deep Learning for Classification},
    author={Kowsari, Kamran and Heidarysafa, Mojtaba and Brown, Donald E. and Jafari Meimandi, Kiana and Barnes, Laura E.},
    booktitle={Proceedings of the 2018 International Conference on Information System and Data Mining},
    year={2018},
    organization={ACM}
    }
