from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
import numpy as np
from utils import *
from model_arch import baseline

data_train, data_dev, data_test = get_data()

vector_size = 100

X_train, Y_train = get_clean_data(data_train, join_sentence = False)
X_dev, Y_dev = get_clean_data(data_dev, join_sentence = False)
X_test, Y_test = get_clean_data(data_test, join_sentence = False)

model_word2vec = Word2Vec(sentences= X_train, vector_size =vector_size, window=5, min_count =10, workers=4, epochs=20)

diction, reverse_diction, counts = get_labels_info(Y_train)

Y_train_num = numerize(Y_train, diction)
Y_dev_num = numerize(Y_dev, diction)
Y_test_num = numerize(Y_test, diction)

pdb.set_trace()

X_train_vec = get_sent_emb(X_train, model_word2vec)
X_dev_vec = get_sent_emb(X_dev, model_word2vec)
X_test_vec = get_sent_emb(X_test, model_word2vec)

model = baseline(nclass = len(diction), shape = vector_size)
file_path = "model_weights/task2_weights.h5"
get_score(model, file_path, X_train_vec, Y_train_num, X_dev_vec, Y_dev_num, X_test_vec, Y_test_num, counts= counts, use_gen=False)
