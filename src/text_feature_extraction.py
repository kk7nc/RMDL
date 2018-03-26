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
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import sys
sys.path.append('../Download_datasets')

import nltk
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk import PorterStemmer
import src.Global as G

cachedStopWords = stopwords.words("english")

def transliterate(line):
    cedilla2latin = [[u'Á', u'A'], [u'á', u'a'], [u'Č', u'C'], [u'č', u'c'], [u'Š', u'S'], [u'š', u's']]
    tr = dict([(a[0], a[1]) for (a) in cedilla2latin])
    new_line = ""
    for letter in line:
        if letter in tr:
            new_line += tr[letter]
        else:
            new_line += letter
    return new_line


def tokenize(text):
  min_length = 3
  words = map(lambda word: word.lower(), word_tokenize(text))
  words = [word for word in words if word not in cachedStopWords]
  tokens = (list(map(lambda token: PorterStemmer().stem(token),
                                   words)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and
                               len(token) >= min_length,
                               tokens))
  return filtered_tokens




def text_cleaner(text, deep_clean=False):
    if deep_clean:
        text = text.replace(".", "")
        text = text.replace("[", " ")
        text = text.replace(",", " ")
        text = text.replace("]", " ")
        text = text.replace("(", " ")
        text = text.replace(")", " ")
        text = text.replace("\"", "")
        text = text.replace("-", " ")
        text = text.replace("=", " ")
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning

        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
        text = text.replace('+', ' ').replace('.', ' ').replace(',', ' ').replace(':', ' ')
        text = re.sub("(^|\W)\d+($|\W)", " ", text)
        text = transliterate(text)
        text = PorterStemmer().stem(text)
    else:
        rules = [
            {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
            {r'\s+': u' '},  # replace consecutive spaces
            {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
            {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
            {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
            {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
            {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
            {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
            {r'^\s+': u''}  # remove spaces at the beginning

        ]
        for rule in rules:
            for (k, v) in rule.items():
                regex = re.compile(k)
                text = regex.sub(v, text)
            text = text.rstrip()
            text = text.strip()
    return text.lower()

def loadData_Tokenizer(X_train, X_test):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test),axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=G.MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=G.MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    #np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train),]
    X_test = text[len(X_train):,]
    embeddings_index = {}
    f = open(G.GloVe_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            print("Warnning"+str(values)+" in" + str(line))
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (X_train, X_test, word_index,embeddings_index)

def W2V_Tokenizer(Data,X_train, X_test):

    nar = [nltk.word_tokenize(sentences) for sentences in Data]
    w2vmodel = gensim.models.Word2Vec(nar, size=100, window=5, min_count=5, workers=4)
    selected_nar = nar
    word_indexes = {}
    for i in range(len(w2vmodel.wv.vocab)):
        word_indexes[w2vmodel.wv.index2word[i]] = i
    narindexed = []

    for a_nar in selected_nar:
        a_nar_indexed = []
        for a_word in a_nar:
            if a_word in word_indexes.keys():
                a_nar_indexed.append(word_indexes[a_word])
            else:
                a_nar_indexed.append(0)
        narindexed.append(a_nar_indexed)

        word_index = w2vmodel.wv.vocab

    data = pad_sequences(narindexed, maxlen=500)

    EMBEDDING_DIM = 100
    embedding_matrix = np.zeros((len(w2vmodel.wv.vocab), 100))
    for i in range(len(w2vmodel.wv.vocab)):
        embedding_vector = w2vmodel.wv[w2vmodel.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    n = 5

    return (X_train, X_test, word_index,embeddings_index)


def loadData(X_train, X_test):
    vectorizer_x = TfidfVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    return (X_train,X_test)