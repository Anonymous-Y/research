# this file shows the structure of the customized deep learning model and how to fine-tune the model

import torch
from torch import nn
import pandas as pd
import numpy as np
import random 
from sklearn.model_selection import train_test_split  
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
import torch.nn.functional as F
import matplotlib.pyplot as plt
import optuna
import wandb
import logging
import spacy
nlp = spacy.load("en_core_web_md")
from sentence_transformers import SentenceTransformer
sbert_trans = SentenceTransformer('all-mpnet-base-v2')

#####################################################################################################################
# load in data
# vote_for: 0 , vote_tighter: 1, vote_easier: 2, vote_other: 9
data = pd.read_feather('parsed_labeled_dataset.ftr')
data = data.loc[data['vote'] != 9]   # remove dissent for other reasons
data = data.loc[data['parsed_words']!=''] # remove rows contain empty string, which means these people did not say anything economy-related
data = data.drop_duplicates()

# replace special chars
data['words']= data['words'].replace(regex=r'\xa0', value =' ')
data['words']= data['words'].replace(regex=r'\xad', value ='-')
data['parsed_words']= data['parsed_words'].replace(regex=r'\xa0', value =' ')
data['parsed_words']= data['parsed_words'].replace(regex=r'\xad', value ='-')

# add title info
# title info: 1 means CHAIR, 2 means VICE CHAIR
title_info = pd.read_feather('title_info.ftr')
data = pd.merge(data, title_info, how='left', on=['meeting', 'name'], sort=False)
data = data.fillna(0)

# remove CHAIR 
data = data.loc[data['title'] != 1]
# we change Burns' title in FOMC19780310confcall to 0
data.loc[(data['meeting']=='FOMC19780310confcall') & (data['name']=='Burns'), 'title'] = 0

# convert 2 to 1, here we only deal with 2 categories
data.loc[data['vote']==2, 'vote'] = 1 

######################################################################################################################
# resample and augment dataset
data = data.sample(frac=1, random_state=28).reset_index(drop=True) 

X_train, X_test, y_train, y_test = train_test_split(data['parsed_words'], data['vote'], test_size=0.2, random_state=42)
X_train = X_train.values.reshape(-1,1)
oversample = RandomOverSampler(sampling_strategy='not majority', random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)

train_data = pd.DataFrame(X_train)
train_data['vote'] = y_train
train_data.columns=['parsed_words', 'vote']

test_data = pd.DataFrame(X_test)
test_data['vote'] = y_test
test_data.columns=['parsed_words', 'vote']

#######################################################################################################################
# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

########################################################################################################################
# create customized dataset
class CustomDataset(Dataset):
    def __init__(self, df_dataset, text_label, vote_label, mat_dim):
        self.rawdata = df_dataset.reset_index(drop=True)
        self.mat_dim = mat_dim
        self.data = []
        for i in range(self.rawdata.shape[0]):
            text = self.rawdata.loc[i, text_label]
            # only keep long sentences
            text = [sent.text.strip() for sent in nlp(text).sents if (sent.text.strip()!= '') & (len(sent.text.strip())>3)]
            # for long text, we only keep the last mat_dim[1] sentences
            text = text[-self.mat_dim[1]:] 
            text = sbert_trans.encode(text)
            vote = self.rawdata.loc[i, vote_label]
            self.data.append([text, vote])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, vote = self.data[idx]
        # pad the dataset
        # method 1: use cv2.resize() funciton to pad 
        #padding_text = cv2.resize(text, self.mat_dim)  # resize take the first dim as col, the second as row, so we need to flip them
        #text_tensor = torch.from_numpy(padding_text)
        # method 2: pad with 0
        text_tensor = torch.from_numpy(text)
        text_pad = self.mat_dim[1] - text.shape[0]
        if text_pad%2==0:
            text_p2d = (0, 0, int(text_pad/2), int(text_pad/2))
        else:
            text_p2d = (0, 0, int((text_pad-1)/2+1), int((text_pad-1)/2))
        text_tensor = F.pad(text_tensor, text_p2d, "constant", 0)

        # ptorch want to read a image as [Channel, Hig, wid], so we need to create an extra dim at the beginning
        text_tensor = torch.unsqueeze(text_tensor, 0)
        class_id = torch.tensor(vote).type(torch.float)  # for binary, this needs to be float
        return text_tensor, class_id

train_tensor = CustomDataset(train_data, 'parsed_words', 'vote', (768, 256))  
test_tensor = CustomDataset(test_data, 'parsed_words', 'vote', (768, 256))

torch.save(train_tensor, 'train_tensor_1overother.pt')
torch.save(test_tensor, 'test_tensor_1overother.pt')

# train_tensor = torch.load('data_gen/train_tensor_1overother.pt')
# test_tensor = torch.load('data_gen/test_tensor_1overother.pt')

BATCH_SIZE = 64
setup_seed(42)
train_dataloader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=True)

#####################################################################################################################
# build the CNN model

# (0) early stopping module
class EarlyStopping():
    """Early stops the training if the target metrics doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='ML_model_parameters.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time the target metrics improved.
                            Default: 7
            verbose (bool): If True, prints a message for each the target metrics improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print 
            mode： 'min' or 'max', it depends on if you want the minimum target mertics or maximum target metrics         
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metrics_update = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def save_checkpoint(self, target_metrics, model):
        '''Saves model when the target metrics changes.'''
        if self.verbose:
            self.trace_func(f'the target metrics updated: ({self.target_metrics_update:.6f} --> {target_metrics:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.target_metrics_update = target_metrics

    def __call__(self, target_metrics, model, mode):
        if mode == 'min':
            score = -target_metrics
        else:
            score = target_metrics

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(target_metrics, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(target_metrics, model)
            self.counter = 0


# (1) Use attention class to build our own class
class residual_block(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.laynorm = nn.LayerNorm(embed_dim)
        self.self_att = nn.MultiheadAttention(embed_dim = embed_dim, num_heads= num_heads, batch_first=True)

    def forward(self, x: torch.Tensor):
        x_res = x
        # apply multi-head self-attention
        x, _ = self.self_att(x, x, x)
        #x = self.laynorm(x)
        x = self.laynorm(x+x_res)
        x = self.dropout(x)
        return x

# genrate a class called FOMCModel and import residual_block class into it
class FOMCModel(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, n_att_mod, dropout_rate):
        super().__init__()
        self.dense = nn.Linear(in_features=768, out_features=embed_dim)
        self.laynorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.self_att_res_mods = nn.ModuleList([residual_block(embed_dim, num_heads, dropout_rate) for _ in range(n_att_mod)])
        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=12288, out_features=embed_dim),
            nn.Dropout(dropout_rate),  # to prevent overfitting
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        x = x.squeeze()
        x = self.dense(x)   # [batch size, seq len, bert embedding] -> [batch size, seq len, embed_dim]
        x = self.laynorm(x)
        x = self.dropout(x)
        for att_res_mod in self.self_att_res_mods:
            x = att_res_mod(x)
        x = self.classifier(x)
        return x


###################################################################################
###################################################################################

def objective(trail):

    params = {
        'n_att_mod': trail.suggest_int('n_att_mod', 1, 12),      # this can pick from 1 to 8 (including 8)
        'num_heads': trail.suggest_int('num_heads', 4, 12, step=4),  # this can pick number 4, 8 or 12, embed_dim must be divisible by num_heads
        'dropout': trail.suggest_float('dropout', 0.2, 0.6, step = 0.005),
        'lr': trail.suggest_float('lr', 1e-6, 1e-3, log=True)  # If log is true, the value is sampled from the range in the log domain.
    }

    setup_seed(42) # set seed to make results reproducible
    fomc_model = FOMCModel(num_classes=1, embed_dim=768, num_heads=params['num_heads'], n_att_mod=params['n_att_mod'], dropout_rate=params['dropout']).to(device)

    # loss fn
    loss_fn = nn.BCEWithLogitsLoss() 
    # Optimizer
    optimizer = torch.optim.Adam(fomc_model.parameters(), lr=params['lr'])
    # make the learning rate adaptive
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    # Metrics
    accuracy_fn = Accuracy(task="binary", threshold = 0.5).to(device)
    precision_fn = Precision(task="binary", threshold = 0.5).to(device)
    recall_fn = Recall(task="binary", threshold = 0.5).to(device)
    # early stop
    path_link = 'C:/Users/Desktop/ML_model_parameters_trail_number_'+str(trail.number)+'.pt'
    early_stop = EarlyStopping(patience= 100, path = path_link, verbose=True)
    best_test_loss = np.inf

    wandb.init(
        # set the wandb project where this run will be logged
        project="FOMC_project",
        name = 'run_' + str(trail.number), # use the trail number info to name the current run
        settings=wandb.Settings(_disable_stats=True, _disable_meta=True), # turn off system metircs logging
        dir='C:/Users/Desktop/',  # set the directory where you want your run to be logged
        # track hyperparameters and run metadata
        config={
        "initial_learning_rate": params['lr'],
        "scheduler": "StepLR_10_95",
        "number_attention_module": params['n_att_mod'],
        "number_attention_head": params['num_heads'],
        "dropout": params['dropout'],
        "early_stop_patience": 50,
        "early_stop_critiria": "min_test_loss",
        "batch_size": 64,
        "dataset": "train_tensor_1overother",
        }
    )

    for epoch in range(1000):
        print(f"Epoch: {epoch}\n-------")
        ### Training
        train_loss, train_acc = 0, 0
        fomc_model.train() 

        # 3. Optimizer zero grad
        optimizer.zero_grad() # clear all the gradients for next epoch

        # Add a loop to loop through training batches
        for batch, (X, y) in enumerate(train_dataloader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_logits = fomc_model(X)
            y_pred = torch.round(torch.sigmoid(y_logits))

            # 2. Calculate loss and accuracy (per batch)
            y = torch.unsqueeze(y, 1)  # add one extra dimension
            loss = loss_fn(y_logits, y)
            train_loss = train_loss + loss # accumulatively add up the loss from batches in the same epoch 
            acc = accuracy_fn(y_pred, y)
            train_acc = train_acc + acc 

            # 4. Loss backward
            loss.backward()

           
        # 5. Optimizer step
        optimizer.step()     # here optimizer will react to each epoch's gradient results 
        
        # Divide total train loss and accuracy by length of train dataloader (average loss per batch per epoch)
        train_loss =  train_loss /len(train_dataloader)
        
        train_acc = train_acc/len(train_dataloader)*100
        train_acc = train_acc.tolist()  # convert to a list

        ### Testing
        # Setup variables for accumulatively adding up loss and accuracy 
        test_loss, test_acc, test_prc, test_rec = 0, 0, 0, 0
        fomc_model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                # Send data to GPU
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_logits = fomc_model(X)
                test_pred = torch.round(torch.sigmoid(test_logits))
                
                # 2. Calculate loss and accuracy (accumatively)
                y = torch.unsqueeze(y, 1)  # add one extra dimension
                test_loss = test_loss + loss_fn(test_logits, y) # accumulatively add up the loss per epoch
                test_acc = test_acc + accuracy_fn(test_pred, y)
                test_prc = test_prc + precision_fn(test_pred, y)
                test_rec = test_rec + recall_fn(test_pred, y)
            
            # Calculations on test metrics need to happen inside torch.inference_mode()
            # Divide total test loss by length of test dataloader (per batch)
            test_loss = test_loss /len(test_dataloader)
            test_acc = test_acc /len(test_dataloader)*100
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss

        ## Print out what's happening
        print(f"\nTrain loss: {train_loss:.5f}, Train acc: {train_acc:.3f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.3f}%, Test prc:{test_prc:.3}%, Test rec:{test_rec:.3}% \n")

        wandb.log({"train loss": train_loss, "train accuracy": train_acc, "test loss": test_loss, "test accuracy": test_acc, "test precision": test_prc, "test recall": test_rec, 'best test loss': best_test_loss})

        # update the learning rate
        scheduler.step()
        print(optimizer.param_groups[0]["lr"]) 

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stop(test_loss, fomc_model, 'min')
        if early_stop.early_stop:
            print('Early stopping reached!')
            wandb.finish()  # finish the wandb run
            break
    
    wandb.finish()  # finish the wandb run
    return best_test_loss


log_name = "ML_model_on_over_other.log"
logger=logging.getLogger() 
logger.setLevel(logging.INFO)  # Setup the root logger.
logger.addHandler(logging.FileHandler(log_name, mode="w"))

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

study = optuna.create_study(direction = 'minimize')
logger.info("Start optimization.")
study.optimize(objective, n_trials=20)

with open(log_name) as f:
    assert f.readline().startswith("A new study created")
    assert f.readline() == "Start optimization.\n"

print('best trial:')