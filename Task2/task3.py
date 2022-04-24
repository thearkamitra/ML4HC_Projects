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

data_train, data_dev, data_test = get_data()

model_name = "emilyalsentzer/Bio_ClinicalBERT"

X_train, Y_train = get_clean_data(data_train, clean_data =False)
X_dev, Y_dev = get_clean_data(data_dev, clean_data =False)
X_test, Y_test = get_clean_data(data_test, clean_data =False)

diction, reverse_diction, counts = get_labels_info(Y_train)

Y_train_num = numerize(Y_train, diction)
Y_dev_num = numerize(Y_dev, diction)
Y_test_num = numerize(Y_test, diction)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case = False)

train_dataloader = create_dataloader(X_train, Y_train_num, tokenizer,)
validation_dataloader = create_dataloader(X_dev, Y_dev_num, tokenizer, train_data=False)

weights = []
for i in range(len(counts)):
    weights.append(counts[i])

model = train_bert(model_name, train_dataloader, validation_dataloader, weights)



prediction_dataloader = create_dataloader(X_test, Y_test_num, tokenizer, train_data = False)
model.eval()

predictions , true_labels = [], []

for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  
  b_input_ids, b_input_mask, b_labels = batch
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)
  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  predictions.append(logits)
  true_labels.append(label_ids)
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = [item for sublist in true_labels for item in sublist]
evaluate(flat_true_labels, flat_predictions)
