
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import re
import numpy as np
stopword_english = list(stopwords.words("english"))
def clean(sentence, stopword_english= stopword_english):
    sentence = sentence.lower()
#     sentence = re.sub(r'[^\w\s]',"",sentence) ##Removes punc from within words
    words = wordpunct_tokenize(sentence)
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    sentence = [x for x in words if x not in stopword_english]
    sentence = [x for x in sentence if x not in punc] #Removes punc and seperates words
    return " ".join(sentence)

def get_clean_data(sentences):
    labels = []
    clean_sentences = []
    for line in sentences:
        line = line.rstrip("\n")
        a = line.split("\t")
        if len(a)<2:
            continue
        label = a[0]
        sentence = a[1]
        sent= clean(sentence)
        labels.append(label)
        clean_sentences.append(sent)
    return clean_sentences, labels


def numerize(Y, diction):
    return [diction[x] for x in Y]

def get_data(data_loc = "Franck-Dernoncourt pubmed-rct master PubMed_200k_RCT"):
    with open(data_loc+"/train.txt","r") as f:
        data_train = f.readlines()
    with open(data_loc+"/dev.txt","r") as f:
        data_dev = f.readlines()
    with open(data_loc+"/test.txt","r") as f:
        data_test = f.readlines()
    return data_train, data_dev, data_test


def get_labels_info(Y_train):
    unq, counts = np.unique(Y_train, return_counts=True)
    diction = {}
    for x in unq:
        diction[x] = len(diction)
    reverse_diction = {v:x for x,v in diction.items()}

    counts = sum(counts)/counts
    counts = counts/np.min(counts)
    return diction, reverse_diction, counts 

