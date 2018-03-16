# Results #

## MNIST annd CIFAR-10 ##

|     Methods    | MNIST | CIFAR-10 |
|:--------------:|:-----:|:--------:|
|   Deep L2-SVM  |  0.87 |   11.9   |
| Maxout Network |  0.94 |   11.68  |
|  BinaryConnect |  1.29 |   9.90   |
|    PCANet-1    |  0.62 |   21.33  |
|    gcForest    |  0.74 |   31.00  |
|  RMDL (3 RDLs) |  0.51 |   9.89   |
|  RMDL (9 RDLs) |  0.41 |    9.1   |
| RMDL (15 RDLs) |  0.21 |   8.74   |
| RMDL (30 RDLs) |  0.18 |   8.79   |


## Web of Science Dataset annd Reuters-21578 ##

|                | WOS-5,736 | WOS-11,967 | WOS-46,985 | Reuters-21578 |
|:--------------:|:---------:|:----------:|:----------:|:-------------:|
|       DNN      |   86.15   |    80.02   |    66.95   |      85.3     |
|       CNN      |   88.68   |    83.29   |    70.46   |      86.3     |
|       RNN      |   89.46   |    83.96   |    72.12   |      88.4     |
|     Naive Bayesian Classifier    |   78.14   |    68.8    |    46.2    |      83.6     |
|       SVM      |   85.54   |    80.65   |    67.56   |      86.9     |
|  SVM (TF-IDF)  |   88.24   |    83.16   |    70.22   |     88.93     |
|  Stacking SVM  |   85.68   |    79.45   |    71.81   |       NA      |
|     HDLTex     |   90.42   |    86.07   |    76.58   |       NA      |
|  RMDL (3 RDLs) |   90.86   |    87.39   |    78.39   |     89.10     |
|  RMDL (9 RDLs) |   92.60   |    90.65   |    81.92   |     90.36     |
| RMDL (15 RDLs) |   92.66   |    91.01   |    81.86   |     89.91     |
| RMDL (30 RDLs) |   93.57   |    91.59   |    82.42   |     90.69     |



## The Olivetti faces dataset ##

|      Methods     | 5 Images | 7 Images | 9 Images |
|:----------------:|:--------:|:--------:|:--------:|
|     gcForest     |   91.00  |   96.67  |   97.50  |
|   Random Forest  |   91.00  |   93.33  |   95.00  |
|        CNN       |   86.50  |   91.67  |   95.00  |
| SVM (rbf kernel) |   80.50  |   82.50  |   85.00  |
|        kNN       |   76.00  |   83.33  |   92.50  |
|        DNN       |   85.50  |   90.84  |   92.5   |
|   RMDL (3 RDL)   |   93.50  |   96.67  |   97.5   |
|   RMDL (9 RDL)   |   93.50  |   98.34  |   97.5   |
|   RMDL (15 RDL)  |   94.50  |   96.67  |   97.5   |
|   RMDL (30 RDL)  |   95.00  |   98.34  |  100.00  |



## 20NewsGroup and IMDB ##


|         Methods         |  IMDB | 20NewsGroup |
|:-----------------------:|:-----:|:-----------:|
|           DNN           | 88.55 |     86.5    |
|           CNN           | 87.44 |    82.91    |
|           RNN           | 88.59 |    83.75    |
|Naive Bayesian Classifier| 83.19 |    81.67    |
|          SVM            | 87.97 |    84.57    |
|       SVM (TF-IDF)      | 88.45 |      86     |
|      RMDL (3 RDLs)      | 89.91 |    86.73    |
|      RMDL (9 RDLs)      | 90.13 |    87.62    |
|     RMDL (15 RDLs)      | 90.79 |    87.91    |
