
from itertools import count
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import re
from tensorflow.keras import optimizers, losses, activations, models, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SimpleRNN, LSTM
from sklearn import metrics
import pdb
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

def get_clean_data(sentences, clean_data = True):
    labels = []
    clean_sentences = []
    for line in sentences:
        line = line.rstrip("\n")
        a = line.split("\t")
        if len(a)<2:
            continue
        label = a[0]
        sentence = a[1]
        if clean_data:
            sentence= clean(sentence)
        labels.append(label)
        clean_sentences.append(sentence)
    return clean_sentences, labels


def numerize(Y, diction):
    return np.asarray([diction[x] for x in Y], dtype= np.int8)

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
    counts = {x:i for x,i in enumerate(counts)}
    return diction, reverse_diction, counts 

def generator(X, Y, BS= 64):
    samples_per_epoch = X.shape[0]
    number_of_batches = samples_per_epoch//BS
    count = 0
    index = np.arange(samples_per_epoch)
    while 1:
        index_batch = index[BS*count:BS*(count+1)]
        X_batch = X[index_batch,:].toarray()
        Y_batch = Y[index_batch]
        count+=1
        yield np.array(X_batch), Y_batch
        if (count>number_of_batches):
            count =0
def get_score(model, file_path, X, Y,  X_dev, Y_dev, X_test, Y_test, counts = None, use_gen = True):

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)
    csvlogger = CSVLogger("Logs/"+file_path.split("/")[1][:-3], append=True, separator = ';')
    callbacks_list = [checkpoint, early, redonplat,csvlogger]  # early
    if counts != None:
        if use_gen:
            aug = generator(X,Y,64)
            model.fit(aug, epochs=1000, verbose=2, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,), class_weight = counts)
        else:
            model.fit(aug, epochs=1000, verbose=2, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,), class_weight = counts)
    else:
        if use_gen:
            aug = generator(X,Y,64)
            model.fit(aug, epochs=1000, verbose=2, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,),)
        else:
            model.fit(X,Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,),)
    model.load_weights(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)
    f1 = metrics.f1_score(Y_test, pred_test, average="weighted")
    print("Test f1 score : %s "% f1)
    acc = metrics.accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s "% acc)
    with open("Logs/"+file_path.split("/")[1][:-3],"a") as f:
        f.write(f"acc= {acc}; f1 = {f1}\n")
    
def get_sent_emb(data, model):
    pass