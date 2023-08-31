# this files shows how to used fine-tuned model to predict the disagreement scores of FOMC transcripts

import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
import torch.nn.functional as F
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("en_core_web_md")
from sentence_transformers import SentenceTransformer
sbert_trans = SentenceTransformer('all-mpnet-base-v2')
# check all the models : https://www.sbert.net/docs/pretrained_models.html


# Setup device-agnostic code 
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


# load in data
# vote_for: 0 , vote_tighter: 1, vote_easier: 2, vote_other: 9
data = pd.read_feather('parsed_labeled_dataset.ftr')
data = data.drop(index = 3797).reset_index(drop=True)  # drop 3797 becuase the only sentence said by Rice is not informative
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

# we change Burns' title in FOMC19780310confcall to 0
data.loc[(data['meeting']=='FOMC19780310confcall') & (data['name']=='Burns'), 'title'] = 0
org_data = data.copy()

##########################################
# convert data to tensor
class CustomDataset_Pred(Dataset):
    def __init__(self, df_dataset, text_label, mat_dim):
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
            self.data.append(text)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        # pad the dataset
        text_tensor = torch.from_numpy(text)
        text_pad = self.mat_dim[1] - text.shape[0]
        if text_pad%2==0:
            text_p2d = (0, 0, int(text_pad/2), int(text_pad/2))
        else:
            text_p2d = (0, 0, int((text_pad-1)/2+1), int((text_pad-1)/2))
        text_tensor = F.pad(text_tensor, text_p2d, "constant", 0)

        # ptorch want to read a image as [Channel, Hig, wid], so we need to create an extra dim at the beginning
        text_tensor = torch.unsqueeze(text_tensor, 0)
        return text_tensor

pred_data = pd.DataFrame({'parsed_words': org_data['parsed_words']})
pred_data = CustomDataset_Pred(pred_data, 'parsed_words', (768, 256))
# torch.save(pred_data, 'pred_org_data_1overother_w_chair_v1.pt')

BATCH_SIZE = 64
# pred_data = torch.load('pred_org_data_1overother_w_chair_v1.pt')
pred_loader = DataLoader(pred_data, batch_size=BATCH_SIZE)

#########################################
# load the model
# (1) Use attention class to build our own class
class FOMCModel(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads):
        super().__init__()

        self.dense = nn.Linear(in_features=768, out_features=embed_dim)

        self.laynorm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(p=0.5)

        self.self_att = nn.MultiheadAttention(embed_dim = embed_dim, num_heads= num_heads, batch_first=True)

        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=12288, out_features=embed_dim),
            nn.Dropout(0.5),  # to prevent overfitting
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        x = x.squeeze()
        x = self.dense(x)   # [batch size, seq len, bert embedding] -> [batch size, seq len, embed_dim]
        x = self.dropout(x)
        x = self.laynorm(x)

        x_res = x
        # apply multi-head self-attention
        x, _ = self.self_att(x, x, x)
        x = self.dropout(x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x_res = x
        x, _ = self.self_att(x, x, x)
        x = self.dropout(x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x_res = x
        x, _ = self.self_att(x, x, x)
        x = self.dropout(x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x_res = x
        x, _ = self.self_att(x, x, x)
        x = self.dropout(x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x_res = x
        x = self.dropout(x)
        x, _ = self.self_att(x, x, x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x_res = x
        x = self.dropout(x)
        x, _ = self.self_att(x, x, x)
        x = self.laynorm(x)
        x = self.laynorm(x+x_res)

        x = self.classifier(x)
        return x


fomc_model = FOMCModel(num_classes=1, embed_dim=768, num_heads=12).to(device)
fomc_model

# load the best model parameters
fomc_model.load_state_dict(torch.load('D:/Dropbox/ML_model_parameters.pt')) # need to load into a machine that supports gpu computing

# Set the model to evaluation mode
fomc_model.eval()
# Predict new data
output_pred_full = []
with torch.inference_mode():
    for X in pred_loader:
            # Send data to GPU
            X = X.to(device)
            output_logits = fomc_model(X)
            output_pred = torch.sigmoid(output_logits)
            output_pred = [x.item() for x in output_pred]
            output_pred_full = output_pred_full + output_pred

org_data['pred'] = output_pred_full
org_data.reset_index(drop=True).to_feather('org_data_w_pred_w_chair_20230407.ftr')
