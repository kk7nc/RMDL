|DOI| |wercker status| |Build Status| |PowerPoint| |license|
|Binder| |pdf| |GitHub license|

RMDL: Random Multimodel Deep Learning for Classification
========================================================

Referenced paper : `RMDL: Random Multimodel Deep Learning for
Classification <www.kowsari.net>`__

Referenced paper : `An Improvement of Data Classification using Random
Multimodel Deep Learning (RMDL) <www.kowsari.net>`__

Random Multimodel Deep Learning (RMDL):
---------------------------------------

A new ensemble, deep learning approach for classification. Deep learning
models have achieved state-of-the-art results across many domains. RMDL
solves the problem of finding the best deep learning structure and
architecture while simultaneously improving robustness and accuracy
through ensembles of deep learning architectures. RDML can accept
asinput a variety data to include text, video, images, and symbolic.


.. image:: http://kowsari.net/onewebmedia/RDL.jpg
    :alt: RDL
    :width: 888 px
    :align: center
    

   

Overview of RDML: Random Multimodel Deep Learning for classification.
The RMDL includesnRandom modelswhich aredrandom model of DNN
classifiers,cmodels of CNN classifiers, andrRNN classifiers
wherer+c+d=n.

Random Multimodel Deep Learning (RDML) architecture for classification.
RMDL includes 3 Random models, oneDNN classifier at left, one Deep CNN
classifier at middle, and one Deep RNN classifier at right (each unit
could be LSTMor GRU).

.. image:: http://kowsari.net/onewebmedia/RMDL.jpg
    :alt: RDL
    :width: 888 px
    :align: center
   

Installation
------------

There are git RMDL in this repository; to clone all the needed files,
please use:

.. code:: python

        - pip install RMDL

Documentation:
--------------

The exponential growth in the number of complex datasets every year
requires more enhancement in machine learning methods to provide robust
and accurate data classification. Lately, deep learning approaches have
been achieved surpassing results in comparison to previous machine
learning algorithms on tasks such as image classification, natural
language processing, face recognition, and etc. The success of these
deep learning algorithms relys on their capacity to model complex and
non-linear relationships between data. However, finding the suitable
structure for these models has been a challenge for researchers. This
paper introduces Random Multimodel Deep Learning (RMDL): a new ensemble,
deep learning approach for classification. RMDL solves the problem of
finding the best deep learning structure and architecture while
simultaneously improving robustness and accuracy through ensembles of
deep learning architectures. In short, RMDL trains multiple models of
Deep Neural Network (DNN), Convolutional Neural Network (CNN) and
Recurrent Neural Network (RNN) in parallel and combines their results to
produce better result of any of those models individually. To create
these models, each deep learning model has been constructed in a random
fashion regarding the number of layers and nodes in their neural network
structure. The resulting RDML model can be used for various domains such
as text, video, images, and symbolic. In this paper, we describe RMDL
model in depth and show the results for image and text classification as
well as face recognition. For image classification, we compared our
model with some of the available baselines using MNIST and CIFAR-10
datasets. Similarly, we used four datasets namely, WOS, Reuters, IMDB,
and 20newsgroup and compared our results with available baselines. Web
of Science (WOS) has been collected by authors and consists of three
sets (small, medium and large set). Lastly, we used ORL dataset to
compare the performance with other face recognition methods. These test
results show that RDML model consistently outperform standard methods
over a broad range of data types and classification problems.

Datasets for RMDL:
------------------

Text Datasets:
~~~~~~~~~~~~~~

-  `IMDB Dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`__

   -  This dataset contains 50,000 documents with 2 categories.

-  `Reters-21578 Dataset <https://keras.io/datasets/>`__

   -  This dataset contains 21,578 documents with 90 categories.

-  `20Newsgroups
   Dataset <https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups>`__

   -  This dataset contains 20,000 documents with 20 categories.

-  Web of Science Dataset (DOI:
   `10.17632/9rw3vkcfy4.2 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__)

   -  Web of Science Dataset
      `WOS-11967 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 11,967 documents with 35 categories which
         include 7 parents categories.

   -  Web of Science Dataset
      `WOS-46985 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 46,985 documents with 134 categories
         which include 7 parents categories.

   -  Web of Science Dataset
      `WOS-5736 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 5,736 documents with 11 categories which
         include 3 parents categories. ### Image datasets: ###

-  `MNIST Dataset <https://en.wikipedia.org/wiki/MNIST_database>`__

   -  The MNIST database contains 60,000 training images and 10,000
      testing images.

-  `CIFAR-10 Dataset <https://www.cs.toronto.edu/~kriz/cifar.html>`__

   -  The CIFAR-10 dataset consists of 60000 32x32 colour images in 10
      classes, with 6000 images per class. There are 50000 training
      images and 10000 test images.

Face Recognition
~~~~~~~~~~~~~~~~

`The Database of Faces (The Olivetti Faces
Dataset) <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`__

-  The The Database of Faces dataset consists of 400 92x112 colour
   images and grayscale in 40 person

Requirment for RMDL :
---------------------

General:

-  Python 3.5 or later see `Instruction
   Documents <https://www.python.org/>`__

-  TensorFlow see `Instruction
   Documents <https://www.tensorflow.org/install/install_linux>`__.

-  scikit-learn see `Instruction
   Documents <http://scikit-learn.org/stable/install.html>`__

-  Keras see `Instruction Documents <https://keras.io/>`__

-  scipy see `Instruction
   Documents <https://www.scipy.org/install.html>`__

-  GPU (if you want to run on GPU):

   -  CUDA® Toolkit 8.0. For details, see `NVIDIA’s
      documentation <https://developer.nvidia.com/cuda-toolkit>`__.

   -  The `NVIDIA drivers associated with CUDA Toolkit
      8.0 <http://www.nvidia.com/Download/index.aspx>`__.

   -  cuDNN v6. For details, see `NVIDIA’s
      documentation <https://developer.nvidia.com/cudnn>`__.

   -  GPU card with CUDA Compute Capability 3.0 or higher.

   -  The libcupti-dev library,

Text and Document Classification
--------------------------------

-  Download GloVe: Global Vectors for Word Representation `Instruction
   Documents <https://nlp.stanford.edu/projects/glove/>`__

   -  Set data directory into
      `Global.py <https://github.com/kk7nc/RMDL/blob/master/src/Global.py>`__

   -  if you are not setting GloVe directory, GloVe will be downloaded

Example
-------

MNIST
~~~~~

.. code:: python

        from keras.datasets import mnist
        import numpy as np
        from RMDL import RMDL_Image as RMDL
        
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train_D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
        X_test_D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
        X_train = X_train_D / 255.0
        X_test = X_test_D / 255.0
        number_of_classes = np.unique(y_train).shape[0]
        shape = (28, 28, 1)
        batch_size = 128
        sparse_categorical = 0
        n_epochs = [10, 500, 50]  ## DNN--RNN-CNN
        Random_Deep = [3, 0, 0]  ## DNN--RNN-CNN
        RMDL.Image_Classification(X_train, y_train, X_test, y_test, batch_size, shape, sparse_categorical, Random_Deep,
                                n_epochs)

Web Of Science
~~~~~~~~~~~~~~

.. code:: python

        from RMDL import text_feature_extraction as txt
        from sklearn.model_selection import train_test_split
        from RMDL.Download import Download_WOS as WOS
        import numpy as np
        from RMDL import RMDL_Text as RMDL

        path_WOS = WOS.download_and_extract()
        fname = os.path.join(path_WOS,"WebOfScience/WOS11967/X.txt")
        fnamek = os.path.join(path_WOS,"WebOfScience/WOS11967/Y.txt")
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
        n_epochs = [5000, 500, 500]  ## DNN--RNN-CNN
        Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN

        RMDL.Text_Classification(X_train, y_train, X_test, y_test, batch_size, sparse_categorical, Random_Deep,
                                n_epochs)

More Exanmple
`link <https://github.com/kk7nc/RMDL/tree/master/Examples>`__ 



Error and Comments:
---------

Send an email to kk7nc@virginia.edu

Citations
---------

::

    @inproceedings{Kowsari2018RMDL,
    title={RMDL: Random Multimodel Deep Learning for Classification},
    author={Kowsari, Kamran and Heidarysafa, Mojtaba and Brown, Donald E. and Jafari Meimandi, Kiana and Barnes, Laura E.},
    booktitle={Proceedings of the 2018 International Conference on Information System and Data Mining},
    year={2018},
    DOI={https://doi.org/10.1145/3206098.3206111},
    organization={ACM}
    }

And

::

    @inproceedings{Heidarysafa2018RMDL,
    title={An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)},
    author={Heidarysafa, Mojtaba and Kowsari, Kamran and  Brown, Donald E. and Jafari Meimandi, Kiana and Barnes, Laura E.},
    booktitle={International Journal of Machine Learning and Computing (IJMLC)},
    year={2018}
    }

.. |DOI| image:: https://img.shields.io/badge/DOI-10.1145/3206098.3206111-blue.svg?style=flat
   :target: https://doi.org/10.1145/3206098.3206111
.. |wercker status| image:: https://app.wercker.com/status/3a564158e809971e2f7416beba5d05af/s/master
   :target: https://app.wercker.com/project/byKey/3a564158e809971e2f7416beba5d05af
.. |Build Status| image:: https://travis-ci.com/kk7nc/RMDL.svg?token=hgKUQ8w7fyzKbCumBbo8&branch=master
   :target: https://travis-ci.com/kk7nc/RMDL
.. |PowerPoint| image:: https://img.shields.io/badge/Presentation-download-red.svg?style=flat
   :target: https://github.com/kk7nc/RMDL/blob/master/Documents/RMDL.pdf
.. |license| image:: https://img.shields.io/badge/ResearchGate-RMDL-blue.svg?style=flat
   :target: https://www.researchgate.net
.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/kk7nc/RMDL/master
.. |pdf| image:: https://img.shields.io/badge/pdf-download-red.svg?style=flat
   :target: https://github.com/kk7nc/RMDL/blob/master/Documents/ACM-RMDL.pdf
.. |GitHub license| image:: https://img.shields.io/badge/licence-GPL-blue.svg
   :target: ./LICENSE
