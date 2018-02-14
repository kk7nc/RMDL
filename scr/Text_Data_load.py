import re
import os
from keras.datasets import imdb
import pandas
from nltk.corpus import stopwords, reuters
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import sys
import numpy as np
import os
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

cachedStopWords = stopwords.words("english")

GLOVE_DIR = "D:/glove/"
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


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    text = string
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
    string = text.lower()

    return string.strip().lower()

def text_cleaner(text):
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
    return text.lower()


def loadData_Tokenizer(X_train, X_test,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):
    np.random.seed(7)
    text = np.concatenate((X_train, X_test),axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    #np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    X_train = text[0:len(X_train),]
    X_test = text[len(X_train):,]
    GLOVE_DIR = "D:/glove/"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding="utf8")
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


def Load_data(Data_text,MAX_NB_WORDS,MAX_SEQUENCE_LENGTH):
    X_train=0
    X_train_M=0
    y_train=0
    X_test=0
    X_test_M=0
    y_test=0
    word_index =0
    embeddings_index =0
    if Data_text ==0:
        fname = "../Data/WOS5736/X.txt"
        fnamek = "../Data/WOS5736/Y.txt"
        with open(fname, encoding="utf-8") as f:
            content = f.readlines()
            content = [clean_str(x) for x in content]
        with open(fnamek) as fk:
            contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
        Label = np.matrix(contentk, dtype=int)
        Label = np.transpose(Label)

        np.random.seed(7)
        print(Label.shape)
        X, X_t, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=4)
        X_train, X_test = loadData(X, X_t)
        X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X, X_t, MAX_NB_WORDS,
                                                                               MAX_SEQUENCE_LENGTH)
    elif (Data_text == 1):

        fname = "../Data/WOS11967/X.txt"
        fnamek = "../Data/WOS11967/Y.txt"
        with open(fname, encoding="utf-8") as f:
            content = f.readlines()
            content = [clean_str(x) for x in content]
        with open(fnamek) as fk:
            contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
        Label = np.matrix(contentk, dtype=int)
        Label = np.transpose(Label)

        np.random.seed(7)
        print(Label.shape)
        X, X_t, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=4)
        X_train, X_test = loadData(X, X_t)
        X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X, X_t, MAX_NB_WORDS,
                                                                               MAX_SEQUENCE_LENGTH)
    elif (Data_text == 2):

        fname = "../Data/WOS46985/X.txt"
        fnamek = "../Data/WOS46985/Y.txt"
        with open(fname, encoding="utf-8") as f:
            content = f.readlines()
            content = [clean_str(x) for x in content]
        with open(fnamek) as fk:
            contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
        Label = np.matrix(contentk, dtype=int)
        Label = np.transpose(Label)

        np.random.seed(7)
        print(Label.shape)
        X, X_t, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=4)
        X_train, X_test = loadData(X, X_t)
        X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X, X_t, MAX_NB_WORDS,
                                                                               MAX_SEQUENCE_LENGTH)
    elif (Data_text == 3):
            documents = reuters.fileids()

            train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                        documents))
            test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                                       documents))
            train_docs = [(reuters.raw(doc_id)) for doc_id in train_docs_id]
            test_docs = [(reuters.raw(doc_id)) for doc_id in test_docs_id]

            mlb = MultiLabelBinarizer()
            y_train = mlb.fit_transform([reuters.categories(doc_id)
                                              for doc_id in train_docs_id])
            y_test = mlb.transform([reuters.categories(doc_id)
                                         for doc_id in test_docs_id])
            y_train = np.argmax(y_train, axis=1)
            y_test = np.argmax(y_test,axis=1)
            print(np.max(y_test))
            print(np.max(y_test))
            X_train, X_test = loadData(train_docs, test_docs)
            X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(train_docs, test_docs, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    elif Data_text == 4:
        TEXT_DATA_DIR = "../Data/20_newsgroup"
        texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        labels = []  # list of label ids
        for name in sorted(os.listdir(TEXT_DATA_DIR)):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                label_id = len(labels_index)
                labels_index[name] = label_id
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        if sys.version_info < (3,):
                            f = open(fpath)
                        else:
                            f = open(fpath, encoding='latin-1')
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                        f.close()
                        labels.append(label_id)

        print('Found %s texts.' % len(texts))

        texts = [text_cleaner(x) for x in texts]
        x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=2)
        X_train, X_test = loadData(x_train, x_test)
        X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(x_train, x_test, MAX_NB_WORDS,
                                                                               MAX_SEQUENCE_LENGTH)
    elif Data_text==5:
            print("Load IMDB dataset....")

            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
            print(len(X_train))
            print(y_test)
            word_index = imdb.get_word_index()
            index_word = {v: k for k, v in word_index.items()}
            X_train = [clean_str(' '.join(index_word.get(w) for w in x)) for x in X_train]
            X_test = [clean_str(' '.join(index_word.get(w) for w in x)) for x in X_test]
            X_train = np.array(X_train)
            X_train = np.array(X_train).ravel()
            print(X_train.shape)
            X_test = np.array(X_test)
            X_test = np.array(X_test).ravel()

            print(np.array(X_test).shape)

            X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X_train, X_test,
                                                                                   MAX_NB_WORDS,
                                                                                   MAX_SEQUENCE_LENGTH)
            X_train, X_test = loadData(X_train, X_test)
    elif Data_text == 6:
            import pandas as pd
            file_x = "D:\CHI\Facebook\X.csv"
            file_y = "D:\CHI\Facebook\X.csv"
            content = pd.read_csv(file_x, encoding="utf-8")
            Label = pd.read_csv(file_y, encoding="utf-8")
            # content = content.as_matrix()
            content = content.ix[:, 1]
            content = np.array(content).ravel()
            print(np.array(content).transpose().shape)
            Label = Label.as_matrix()
            Label = np.matrix(Label)
            np.random.seed(7)
            # print(Label)
            content = [text_cleaner(x) for x in content]
            X, X_t, y_train, y_test = train_test_split(content, Label, test_size=0.1, random_state=0)
            X_train, X_test = loadData(X, X_t)
            X_train_M, X_test_M, word_index, embeddings_index = loadData_Tokenizer(X, X_t, MAX_NB_WORDS,
                                                                                   MAX_SEQUENCE_LENGTH)

    number_of_classes = np.max(y_train)+1
    return (X_train,X_train_M, y_train,X_test, X_test_M, y_test, word_index, embeddings_index, number_of_classes)


def loadData(X_train, X_test):
    vectorizer_x = TfidfVectorizer()
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print(np.array(X_train).shape)
    return (X_train,X_test)