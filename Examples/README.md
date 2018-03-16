# Results #

## Image Classification ##
- [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database)

  * The MNIST database contains 60,000 training images and 10,000 testing images.
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

  * The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
  
### MNIST annd CIFAR-10 ###

|     Methods    | MNIST | CIFAR-10 |
|:----------------:|:-----:|:--------:|
|   Deep L2-SVM    |  0.87 |   11.9   |
| Maxout Network   |  0.94 |   11.68  |
|  BinaryConnect   |  1.29 |   9.90   |
|    PCANet-1      |  0.62 |   21.33  |
|    gcForest      |  0.74 |   31.00  |
|**RMDL (3 RDLs)** |  0.51 |   9.89   |
|**RMDL (9 RDLs)** |  0.41 |    9.1   |
|**RMDL (15 RDLs)**|  0.21 |   8.74   |
|**RMDL (30 RDLs)**|  0.18 |   8.79   |

## Text Classification ##

- [Reters-21578 Dataset](https://keras.io/datasets/)

  * This dataset contains 21,578 documents with 90 categories.
      
  
- Web of Science Dataset (DOI: [10.17632/9rw3vkcfy4.2](http://dx.doi.org/10.17632/9rw3vkcfy4.2))


### Web of Science Dataset annd Reuters-21578 ###

|                                               | WOS-5,736 | WOS-11,967 | WOS-46,985 | Reuters-21578 |
|:---------------------------------------------:|:---------:|:----------:|:----------:|:-------------:|
|          Deep Neural Networks (DNN)           |   86.15   |    80.02   |    66.95   |      85.3     |
|      Convolutional Neural Netwroks (CNN)      |   88.68   |    83.29   |    70.46   |      86.3     |
|       Recurrent Neural Networks (DNN)         |   89.46   |    83.96   |    72.12   |      88.4     |
|           Naive Bayesian Classifier           |   78.14   |    68.8    |    46.2    |      83.6     |
|         Support Vector Machine (SVM)          |   85.54   |    80.65   |    67.56   |      86.9     |
|  Support Vector Machine (SVM using TF-IDF)    |   88.24   |    83.16   |    70.22   |     88.93     |
|         Stacking Support Vector Machine       |   85.68   |    79.45   |    71.81   |       NA      |
|                  HDLTex                       |   90.42   |    86.07   |    76.58   |       NA      |
|               RMDL (3 RDLs)                   |   90.86   |    87.39   |    78.39   |     89.10     |
|               RMDL (9 RDLs)                   |   92.60   |    90.65   |    81.92   |     90.36     |
|               RMDL (15 RDLs)                  |   92.66   |    91.01   |    81.86   |     89.91     |
|               RMDL (30 RDLs)                  |   93.57   |    91.59   |    82.42   |     90.69     |

### 20NewsGroup and IMDB ###

- [20Newsgroups Dataset](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)

  * This dataset contains 20,000 documents with 20 categories.
  
- [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

  * This dataset contains 50,000 documents with 2 categories.


|         Methods                          |  IMDB | 20NewsGroup |
|:----------------------------------------:|:-----:|:-----------:|
|          Deep Neural Networks (DNN)      | 88.55 |     86.5    |
|   Convolutional Neural Netwroks (CNN)    | 87.44 |    82.91    |
|    Recurrent Neural Networks (RNN)       | 88.59 |    83.75    |
|    Naive Bayesian Classifier (NBC)       | 83.19 |    81.67    |
|         Support Vector Machine (SVM)     | 87.97 |    84.57    |
|Support Vector Machine (SVM using TF-IDF) | 88.45 |      86     |
|            RMDL (3 RDLs)                 | 89.91 |    86.73    |
|            RMDL (9 RDLs)                 | 90.13 |    87.62    |
|            RMDL (15 RDLs)                | 90.79 |    87.91    |


## Face Recognition ##

[The Database of Faces (The Olivetti Faces Dataset)](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

   * he files are in PGM format, and can conveniently be viewed on UNIX (TM) systems using the 'xv' program. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).
   
### The Olivetti Faces Dataset ###

|      Methods                   | 5 Images | 7 Images | 9 Images |
|:------------------------------:|:--------:|:--------:|:--------:|
|          gcForest           |   91.00  |   96.67  |   97.50  |
|        Random Forest        |   91.00  |   93.33  |   95.00  |
|Convolutional Neural Netwroks|   86.50  |   91.67  |   95.00  |
|       SVM (rbf kernel)      |   80.50  |   82.50  |   85.00  |
|   k-nearest neighbors (kNN) |   76.00  |   83.33  |   92.50  |
| Deep Neural Networks (DNN)  |   85.50  |   90.84  |   92.5   |
|        RMDL (3 RDL)         |   93.50  |   96.67  |   97.5   |
|        RMDL (9 RDL)         |   93.50  |   98.34  |   97.5   |
|        RMDL (15 RDL)        |   94.50  |   96.67  |   97.5   |
|        RMDL (30 RDL)        |   95.00  |   98.34  |  100.00  |



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
    
    
