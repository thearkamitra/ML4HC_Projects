
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
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import DistilBertModel, DistilBertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import time
import datetime
import random
import os
import os.path
import numpy as np
import string
stopword_english = list(stopwords.words("english"))
def clean(sentence, stopword_english= stopword_english, join_sentence=True):
    sentence = sentence.lower()
#     sentence = re.sub(r'[^\w\s]',"",sentence) ##Removes punc from within words
    words = wordpunct_tokenize(sentence)
    punc = string.punctuation
    sentence = [x for x in words if x not in stopword_english]
    sentence = [x for x in sentence if x not in punc] #Removes punc and seperates words
    if join_sentence:
        return " ".join(sentence)
    return sentence

def get_clean_data(sentences, clean_data = True, join_sentence=True):
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
            sentence= clean(sentence, join_sentence= join_sentence)
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
def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))
def get_score(model, file_path, X, Y,  X_dev, Y_dev, X_test, Y_test, counts = None, use_gen = True):

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=3, verbose=2)
    csvlogger = CSVLogger("Logs/"+file_path.split("/")[1][:-3], append=True, separator = ';')
    callbacks_list = [checkpoint, early, redonplat,csvlogger]  # early
    if counts != None:
        if use_gen:
            aug = generator(X,Y,200)
            model.fit(aug, epochs=1000, verbose=1, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,), class_weight = counts)
        else:
            model.fit(X, Y , epochs=1000, verbose=1, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,), class_weight = counts)
    else:
        if use_gen:
            aug = generator(X,Y,200)
            model.fit(aug, epochs=1000, verbose=1, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,),)
        else:
            model.fit(X, Y, epochs=1000, verbose=1, callbacks=callbacks_list, validation_data = (X_dev,Y_dev,),)
    model.load_weights(file_path)
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)
    f1 = metrics.f1_score(Y_test, pred_test, average="weighted")
    print("Test f1 score : %s "% f1)
    acc = metrics.accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s "% acc)
    with open("Logs/"+file_path.split("/")[1][:-3],"a") as f:
        f.write(f"acc= {acc}; f1 = {f1}\n")
    
def get_sent_emb(data, model, vector_size = 100):
    sent_embs = []
    for sent in data:
        sent_emb = []
        for word in sent:
            try:
                sent_emb.append(model.wv[word])
            except:
                pass
        if len(sent_emb)==0:
            sent_emb = np.zeros(100,) 
        else:
            sent_emb = np.asarray(sent_emb)
            sent_emb = np.mean(sent_emb,axis =0)
        sent_embs.append(sent_emb)
    return np.asarray(sent_embs)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def evaluate(y_true, y_pred):
    
    assert len(y_true) == len(y_pred)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average='weighted')
    scores = [
        "F1: %f\n" % f1,
        "Recall: %f\n" % recall,
        "Precision: %f\n" % precision,
        "ExactMatch: %f\n" % -1.0
    ]
    for s in scores:
        print(s, end='')

def create_dataloader(X_data, Y_data, tokenizer, MAX_LEN=200, batch_size=32, train_data= True):
    input_ids = []
    for sent in X_data:
        encoded_sent = tokenizer.encode(str(sent),add_special_tokens = True,)
        input_ids.append(encoded_sent)
    print("Tokenized")
    pad_token = tokenizer.pad_token_id
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=pad_token, truncating="post", padding="post")
    print("Padded")
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id != pad_token) for token_id in sent]
        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    print("Generated masks")
    inputs = input_ids
    labels = Y_data
    masks = attention_masks
    inputs = torch.tensor(inputs).to(torch.int64)
    masks = torch.tensor(masks)
    labels = torch.tensor(labels).to(torch.int64)
    print("Converted to tensors")
    data = TensorDataset(inputs, masks, labels)
    if train_data:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    print("Converted to data")
    dataloader = DataLoader(data, sampler= sampler, batch_size = batch_size)
    print("Created Dataloader")
    return dataloader


def train_bert(model_name, train_dataloader, validation_dataloader, weights):
        
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions = False,output_hidden_states = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = list(model.named_parameters())

    optimizer = AdamW(model.parameters(),
                    lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    epochs = 4

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    seed_val = 42


    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []
    criterion = nn.CrossEntropyLoss(weight = torch.Tensor(weights).to(device))
    # For each epoch...
    for epoch_i in range(0, epochs):

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device).long()
            model.zero_grad()        
            outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
            
            logits = outputs[0]
            loss = criterion(logits,b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)            
        
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
            
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            
            batch = tuple(t.to(device) for t in batch)
            
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():        

                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_accuracy += tmp_eval_accuracy

            nb_eval_steps += 1

        print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    return model