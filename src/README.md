Build Status:
[![wercker status](https://app.wercker.com/status/3a564158e809971e2f7416beba5d05af/s/master "wercker status")](https://app.wercker.com/project/byKey/3a564158e809971e2f7416beba5d05af)
[![GitHub license](https://img.shields.io/badge/licence-GPL-blue.svg)](./LICENSE)


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

        * To install this library, issue the following command:

                 $ sudo apt-get install libcupti-dev
               

## Error and Comments: ##

Send an email to [kk7nc@virginia.edu](mailto:kk7nc@virginia.edu)


## Citations ##

    @inproceedings{Kowsari2018RMDL,
    title={RMDL: Random Multimodel Deep Learning for Classification},
    author={Kowsari, Kamran and Heidarysafa, Mojtaba and Brown, Donald E. and Jafari Meimandi, Kiana and Barnes, Laura E.},
    booktitle={Proceedings of the 2018 International Conference on Information System and Data Mining},
    year={2018},
    organization={ACM}
    }

And

    @inproceedings{Heidarysafa2018RMDL,
    title={An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)},
    author={Heidarysafa, Mojtaba and Kowsari, Kamran and  Brown, Donald E. and Jafari Meimandi, Kiana and Barnes, Laura E.},
    booktitle={International Journal of Machine Learning and Computing (IJMLC)},
    year={2018}
    }

    
