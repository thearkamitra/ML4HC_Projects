from curses import window
from tkinter import W
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
import numpy as np
from utils import *

data_train, data_dev, data_test = get_data()


X_train, Y_train = get_clean_data(data_train)
X_dev, Y_dev = get_clean_data(data_dev)
X_test, Y_test = get_clean_data(data_test)

model = Word2Vec(sentences= X_train, vector_size =100, window=5, min_count =10, workers=4, epochs=20)

diction, reverse_diction, counts = get_labels_info(Y_train)

Y_train_num = numerize(Y_train, diction)
Y_dev_num = numerize(Y_dev, diction)
Y_test_num = numerize(Y_test, diction)

