# [RMDL: Random Multimodel Deep Learning for Classification]()

All codes and Datasets: coming soon


Refrenced paper : [RMDL: Random Multimodel Deep Learning for Classification]()


Refrenced paper : [An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)]()


**Random  Multimodel  Deep  Learn-ing (RMDL):**   a   new   ensemble,   deep   learning   approach   for classification.  Deep  learning  models  have  achieved  state-of-the-art  results  across  many  domains.  RMDL  solves  the  problem of  finding  the  best  deep  learning  structure  and  architecture while simultaneously improving robustness and accuracy through ensembles  of  deep  learning  architectures.  RDML  can  accept  asinput a variety data to include text, video, images, and symbolic. 


Overview  of  RDML:  Random  Multimodel  Deep  Learning  for  classification.  The  RMDL  includesnRandom  modelswhich aredrandom model of DNN classifiers,cmodels of CNN classifiers, andrRNN classifiers wherer+c+d=n.


<p align="center">
<img src="http://kowsari.net/onewebmedia/RDL.jpg" width="80%"></img> 
</p>


Random  Multimodel  Deep  Learning  (RDML)  architecture  for  classification.  RMDL  includes  3  Random  models,  oneDNN classifier at left, one Deep CNN classifier at middle, and one Deep RNN classifier at right (each unit could be LSTMor GRU).


<p align="center">
<img src="http://kowsari.net/onewebmedia/RMDL.jpg" width="80%"></img> 
</p>


**Documentation:**
The continually increasing number of complex datasets each year necessitates ever improving machine learning methods for robust and accurate categorization of these data. This paper introduces Random Multimodel Deep Learning~(RMDL): a new ensemble, deep learning approach for classification. Deep learning models have achieved state-of-the-art results across many domains.  RMDL solves the problem of finding the best deep learning structure and architecture while simultaneously improving robustness and accuracy through ensembles of deep learning architectures. RDML can accept as input a variety data to include text, video, images, and symbolic. This paper describes RMDL and shows test results for image and text data including MNIST, CIFAR-10, WOS, Reuters, IMDB, and 20newsgroup. These test results show that RDML produces consistently better performance than standard methods over a broad range of data types and classification problems.

**Datasets for RMDL:** 

**Text Datasets:** 

[IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

        This dataset contains 50,000 documents with 2 categories.
[Reters-21578 Dataset](https://keras.io/datasets/)

        This dataset contains 21,578 documents with 90 categories.
[20Newsgroups Dataset](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)

        This dataset contains 20,000 documents with 20 categories.
Web of Science Dataset [WOS-11967](https://github.com/kk7nc/HDLTex/tree/master/DATA)

        This dataset contains 11,967 documents with 35 categories which include 7 parents categories.
Web of Science Dataset [WOS-46985](https://github.com/kk7nc/HDLTex/tree/master/DATA)

        This dataset contains 46,985 documents with 134 categories which include 7 parents categories.
      
Web of Science Dataset [WOS-5736](https://github.com/kk7nc/HDLTex/tree/master/DATA)

        This dataset contains 5,736 documents with 11 categories which include 3 parents categories.

**Image datasets:** 

[MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)

        The MNIST database contains 60,000 training images and 10,000 testing images.
[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

        The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
**Requirment for RMDL :**


General:

Python 3.5 or later see [Instruction Documents](https://www.python.org/)

TensorFlow see [Instruction Documents](https://www.tensorflow.org/install/install_linux).

scikit-learn see [Instruction Documents](http://scikit-learn.org/stable/install.html)

Keras see [Instruction Documents](https://keras.io/)

scipy see [Instruction Documents](https://www.scipy.org/install.html)

GPU (if you want to run on GPU):

CUDA® Toolkit 8.0. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cuda-toolkit). 

The [NVIDIA drivers associated with CUDA Toolkit 8.0](http://www.nvidia.com/Download/index.aspx).

cuDNN v6. For details, see [NVIDIA's documentation](https://developer.nvidia.com/cudnn). 

GPU card with CUDA Compute Capability 3.0 or higher.

The libcupti-dev library,

To install this library, issue the following command:

```
$ sudo apt-get install libcupti-dev
```

