import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils import *
from model_arch import baseline
from sklearn import metrics
classifier = "keras"
vectorizer = TfidfVectorizer(max_features = 2000)
naive_bayes_classifier = MultinomialNB()
data_train, data_dev, data_test = get_data()
if classifier=="bayes":
    data_train = data_train + data_dev

X_train, Y_train = get_clean_data(data_train)
X_dev, Y_dev = get_clean_data(data_dev)
X_test, Y_test = get_clean_data(data_test)

X_train_tf_idf = vectorizer.fit_transform(X_train)
X_dev_tf_idf = vectorizer.transform(X_dev)
X_test_tf_idf = vectorizer.transform(X_test)

diction, reverse_diction, counts = get_labels_info(Y_train)

Y_train_num = numerize(Y_train, diction)
Y_dev_num = numerize(Y_dev, diction)
Y_test_num = numerize(Y_test, diction)


if classifier =="keras":
    model = baseline(nclass = len(diction), shape = X_train_tf_idf.shape[-1])
    
elif classifier =="bayes":
    naive_bayes_classifier.fit(X_train_tf_idf, Y_train_num)
    y_pred = naive_bayes_classifier.predict(X_test_tf_idf)
    f1_score = metrics.f1_score(Y_train_num, y_pred, average="weighted")
    print(f1_score)