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

model_name = "bert-large-cased"

X_train, Y_train = get_clean_data(data_train, clean_data =False)
X_dev, Y_dev = get_clean_data(data_dev, clean_data =False)
X_test, Y_test = get_clean_data(data_test, clean_data =False)

diction, reverse_diction, counts = get_labels_info(Y_train)

Y_train_num = numerize(Y_train, diction)
Y_dev_num = numerize(Y_dev, diction)
Y_test_num = numerize(Y_test, diction)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name,do_lower_case = False)

input_ids = []
dev_input_ids = []
test_input_ids = []
for sent in X_train:
    encoded_sent = tokenizer.encode(str(sent),add_special_tokens = True,)
    input_ids.append(encoded_sent)

# For every sentence...
for sent in X_dev:
    encoded_sent = tokenizer.encode(str(sent),add_special_tokens = True,)
    dev_input_ids.append(encoded_sent)


for sent in X_test:
    encoded_sent = tokenizer.encode(str(sent),add_special_tokens = True,)
    test_input_ids.append(encoded_sent)

MAX_LEN = 200

print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)

print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

# Pad our input tokens with value 0.
# "post" indicates that we want to pad and truncate at the end of the sequence,
# as opposed to the beginning.
pad_token = tokenizer.pad_token_id

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=pad_token, truncating="post", padding="post")

dev_input_ids = pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long", 
                          value=pad_token, truncating="post", padding="post")

print('\nDone.')

# Create attention masks
attention_masks = []
dev_attention_masks = []
test_attention_masks = []
# For each sentence...
for sent in input_ids:
    att_mask = [int(token_id != pad_token) for token_id in sent]
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

# For each sentence...
for sent in dev_input_ids:
    att_mask = [int(token_id != pad_token) for token_id in sent]
    # Store the attention mask for this sentence.
    dev_attention_masks.append(att_mask)


for sent in test_input_ids:
    att_mask = [int(token_id != pad_token) for token_id in sent]
    # Store the attention mask for this sentence.
    test_attention_masks.append(att_mask)
    

# Use 90% for training and 10% for validation.
train_inputs = input_ids 
validation_inputs = dev_input_ids
test_inputs = test_input_ids
train_labels = Y_train_num
validation_labels = Y_dev_num
test_labels = Y_test_num

# Do the same for the masks.
train_masks =  attention_masks
validation_masks = dev_attention_masks
test_masks = test_attention_masks                                             
# Convert all inputs and labels into torch tensors, the required datatype 
# for our model.
train_inputs = torch.tensor(train_inputs).to(torch.int64)
validation_inputs = torch.tensor(validation_inputs).to(torch.int64)

train_labels = torch.tensor(train_labels).to(torch.int64)
validation_labels = torch.tensor(validation_labels).to(torch.int64)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size = 16

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


test_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
test_sampler = SequentialSampler(validation_data)
test_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions = False,output_hidden_states = False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#model.cuda()

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The XLNet model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

epochs = 4

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
                                            

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
loss_values = []
weights = []
for i in range(len(counts)):
    weights.append(counts[i])
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

prediction_inputs = torch.tensor(test_input_ids).to(torch.int64)
prediction_masks = torch.tensor(test_attention_masks)
prediction_labels = torch.tensor(test_labels).to(torch.int64)

# Set the batch size.  
batch_size = 8  

# Create the DataLoader.
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  batch = tuple(t.to(device) for t in batch)
  
  b_input_ids, b_input_mask, b_labels = batch
  with torch.no_grad():
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')
with open('predictionstrue.txt', "w") as writer:
    for i,line in enumerate(predictions):
        writer.write(str(line) +" " +str(true_labels[i]) + "\n")
  
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

flat_true_labels = [item for sublist in true_labels for item in sublist]

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

# Evaluate predictions    
evaluate(flat_true_labels, flat_predictions)

print('Writing predictions to file...')

# Save predictions to file
with open('predictions.txt', "w") as writer:
    for line in flat_predictions:
        writer.write(str(line) + "\n")
        
print('Done writing predictions...')