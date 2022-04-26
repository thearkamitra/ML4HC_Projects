import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from transformers import DistilBertModel, DistilBertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch
import torch.nn as nn
import time
import datetime
import random
import os
import os.path
from sklearn import metrics
from utils import *
from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-w","--weights",type = int, choices = [0,1], \
    help="Whether to consider unbalanced weights or not.", default=1)
args = parser.parse_args()

diction = {'BACKGROUND': 0, 'CONCLUSIONS': 1, 'METHODS': 2, 'OBJECTIVE': 3, 'RESULTS': 4}
rev_diction = {0: 'BACKGROUND', 1: 'CONCLUSIONS', 2: 'METHODS', 3: 'OBJECTIVE', 4: 'RESULTS'}
counts = {0: 3.8958508101622358,
 1: 2.255635622906327,
 2: 1.0604564716172191,
 3: 4.106467810997797,
 4: 1.0}

dataset = load_dataset(path = "Franck-Dernoncourt pubmed-rct master PubMed_200k_RCT",data_files = {'train':"train.txt", 'dev' :'dev.txt', 'test':'test.txt'}, )
dataset = dataset.filter(lambda x: len(x['text'].split('\t'))>=2)
dataset = dataset.map(lambda x: {'label': diction[x['text'].split('\t')[0]] , 'text': x['text'].split('\t')[1]} )

model_name = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case = False, model_max_length=512, padding_side="right", truncation_side="right")

def tokenize_function(sentences):
    return tokenizer(sentences['text'], padding = 'max_length', max_length =200)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

metric = load_metric("f1", average="weighted")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

train_dataset = tokenized_dataset["train"]
dev_dataset = tokenized_dataset["dev"]
test_dataset = tokenized_dataset["test"]

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", \
    load_best_model_at_end = True, do_train=False)


class_weights = [x for a,x in counts.items()]

if args.weights==0:
    class_weights = [1 for x in class_weights]

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)


trainer.train()
