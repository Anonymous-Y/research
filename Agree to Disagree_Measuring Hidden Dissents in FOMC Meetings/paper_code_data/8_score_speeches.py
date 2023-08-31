import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_md")
from sentence_transformers import SentenceTransformer
#sbert_trans = SentenceTransformer('all-MiniLM-L6-v2')
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
data = pd.read_feather('speech_data/speech_data_full.ftr')

##########################################
# parse the data to keep sentences that contain econ-related phrases

# (a) generate the econ-related phrases
term1 = pd.read_feather('oxford_dic_econ.ftr')
term2 = pd.read_feather('oxford_dic_fb.ftr')
#term3 = pd.read_feather('sentiment_dict.ftr')   # this data set is too large.

term1 = term1.loc[1:,]  # remove the term 1992
term1_cap = term1['terms'].str.capitalize()
term1 = term1['terms'].to_list() + term1_cap.to_list()

term2_cap = term2['terms'].str.capitalize()
term2 = term2['terms'].to_list() + term2_cap.to_list()

phrases = term1 + term2
phrases = list(set(phrases)) # drop duplicated items

# drop some common words, like or, will
phrases = [x for x in phrases if x !='or' and x !='will' and x !='Or' and x != 'Will' and x != 'OR' and x !='WILL']

# only keep related sentences
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(text) for text in phrases]
phrase_matcher.add('econ', patterns)

for i in range(data.shape[0]):
    doc = nlp(data.loc[i, 'speech'])
    # extract the related sentences
    temp = [sent.text.strip() for sent in doc.sents if phrase_matcher(nlp(sent.text.strip())) != []]
    # remove redundant empty space
    temp = [re.sub('\s{2,100}', ' ', x) for x in temp]
    # check the # of words in each sentence, then delete the ones that contain less than 5 words.
    temp = pd.DataFrame({'speech': temp})
    temp['length'] = temp['speech'].apply(lambda x: len(x.split()))
    temp = temp.loc[temp['length']>5]
    # combine a list of string into one string
    data.loc[i, 'parsed_speech'] = ' '.join(temp['speech'])
    data.loc[i, 'num_of_sents'] = temp.shape[0]
    if i % 100 == 0:
        print (i, 'is done.\n')


data.reset_index(drop=True).to_feather('speech_data/parsed_speech.ftr')
data.to_csv('speech_data/parsed_speech.csv', index=False)

data['num_of_sents'].describe()


#############################################################################
# chunk the original data
# cut long speech into chunks
chunk_data = data[['link', 'parsed_speech', 'num_of_sents']].copy()

link = []
chunk_id = []
chunk_parsed_speech = []
for i in range(chunk_data.shape[0]):
    doc = nlp(chunk_data.loc[i, 'parsed_speech'])
    temp = [sent.text.strip() for sent in doc.sents]
    text_len = len(temp)
    n = 0
    while text_len >0:
        link.append(chunk_data.loc[i, 'link'])
        chunk_id.append(n)
        chunk_temp = temp[n*256: (n+1)*256]
        chunk_parsed_speech.append(' '.join(chunk_temp))
        n = n+1
        text_len = text_len - 256
    
    if i % 100 == 0:
        print (i, 'is done.\n')

data_chunks = pd.DataFrame({'link': link, 'chunk_id': chunk_id, 'chunk_parsed_speech': chunk_parsed_speech})
data_chunks.to_feather('speech_data/parsed_speech_chunks.ftr')

##########################################
# convert data to tensor
class CustomDataset_Pred(Dataset):
    def __init__(self, df_dataset, text_label, mat_dim):
        self.rawdata = df_dataset.reset_index(drop=True)
        self.mat_dim = mat_dim
        self.data = []
        for i in range(self.rawdata.shape[0]):
            text = self.rawdata.loc[i, text_label]
            #text = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()!= '']
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

pred_speech = pd.DataFrame({'chunk_parsed_speech': data_chunks['chunk_parsed_speech']})
pred_speech = CustomDataset_Pred(pred_speech, 'chunk_parsed_speech', (768, 256))
# torch.save(pred_speech, 'speech_data/pred_all_speech_data_1overother_v1.pt')  

BATCH_SIZE = 64
# pred_speech = torch.load('speech_data/pred_all_speech_data_1overother_v1.pt')
pred_loader = DataLoader(pred_speech, batch_size=BATCH_SIZE)

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

data_chunks['pred'] = output_pred_full
data_chunks.reset_index(drop=True).to_feather('speech_data/speech_chunks_with_pred.ftr')
