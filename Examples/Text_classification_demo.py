"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
RMDL: Random Multimodel Deep Learning for Classification

* Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
* Last Update: Oct 26, 2018
* This file is part of  RMDL project, University of Virginia.
* Free to use, change, share and distribute source code of RMDL
* Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
* Link: https://dl.acm.org/citation.cfm?id=3206111
* Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
* Link :  http://www.ijmlc.org/index.php?m=content&c=index&a=show&catid=79&id=823
* Comments and Error: email: kk7nc@virginia.edu

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#A very simple and minimal demo to show the use of Classification model ,
#It will clear the doubt how to format the data for model

#importing libraries
import sys
import os
from RMDL import text_feature_extraction as txt
from keras.datasets import imdb
import numpy as np
from RMDL import RMDL_Text as RMDL



#Data description

#sentences should be in this format:
sentences=['everyone please come check our newest song in memories of Martin Luther  King Jr', 'Came here to check the views, goodbye.', 'sub my channel for no reason.', 'Check out my dubstep song "Fireball", made with Fruity Loops. I really took  time in it.', '2 billion Coming soon', 'Why dafuq is a Korean song so big in the USA. Does that mean we support  Koreans? Last time I checked they wanted to bomb us.', 'Check my channel please! And listen to the best music ever ', 'SUB 4 SUB PLEASE LIKE THIS COMMENT I WANT A SUCCESFULL YOUTUBE SO PPLEASE LIKE THIS  COMMENT AND SUBSCRIBE IT ONLY TAKES 10 SECONDS PLEASE IF YOU SUBSCRIBE ILL  SUBSCRIBE BACK THANKS', ' Hey everyone!! I have just started my first YT channel i would be grateful  if some of you peoples could check out my first clip in BF4! and give me  some advice on how my video was and how i could improve it. ALSO be sure to  go check out the about to see what Im all about. Thanks for your time :) .  and to haters. You Hate, I WIN', 'The projects After Effects, Music, Foto, Web sites and another you can find  and buy here']

#labels can be one hot encoded or like this:
labels = [1, 0, 1, 1, 0, 0, 1, 1, 1, 1]


#is your labels are like this then you can directly feed to network

#After formatting your data like above data Let's use the network and feed the data


#split the data into train and test dataset

print(len(sentences))
split_data = int(len(sentences) * 0.85)

train_sentences = sentences[:split_data]
train_labels = labels[:split_data]

test_sentences = sentences[split_data:]
test_labels = labels[split_data:]


#batch_size should not be very small neither too big
batch_size = 2


sparse_categorical = 0

#epoch for DNN , RNN and CNN
n_epochs = [5, 5, 5]  ## DNN--RNN-CNN
Random_Deep = [3, 3, 3]  ## DNN--RNN-CNN
no_of_classes = 2
RMDL.Text_Classification(np.array(train_sentences), np.array(train_labels), np.array(test_sentences),
                         np.array(test_labels),
                         batch_size=batch_size,
                         sparse_categorical=sparse_categorical,
                         random_deep=Random_Deep,
                         epochs=n_epochs, no_of_classes=2)


#output
#
# Found 129 unique tokens.
# (10, 500)
# Total 400000 word vectors.
# 2
# DNN 0
# <keras.optimizers.Adagrad object at 0x7f00801bbb70>
# Train on 8 samples, validate on 2 samples
# Epoch 1/5
#  - 0s - loss: 0.8781 - acc: 0.5000 - val_loss: 0.1762 - val_acc: 1.0000
#
# Epoch 00001: val_acc improved from -inf to 1.00000, saving model to weights\weights_DNN_0.hdf5
# Epoch 2/5
#  - 0s - loss: 0.9983 - acc: 0.7500 - val_loss: 0.0240 - val_acc: 1.0000
