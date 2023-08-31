import pandas as pd
import re
import os

############################################################################################
# Part I, prepare the raw labeled data set 
voting_table = pd.read_csv('voting_results.csv')

root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/Tsang_Yang_Research/FOMC/parsed_raw_data/'

meeting_list = voting_table['meeting'].drop_duplicates().reset_index(drop=True)

labeled_data = pd.DataFrame()
for i in range(meeting_list.shape[0]):
    data = pd.read_feather(file_path+meeting_list[i]+'.ftr')
    data['name'] = data['name'].str.replace(r'(CHAIR|VICE|MR\.|MRS\.|MS\.)', '', regex=True)
    data['name'] = data['name'].str.strip()  # remove extra empty space
    data['name'] = data['name'].str.capitalize() # convert the names to be first letter uppercase
    data = data.groupby(['name'])['words'].apply(' '.join).reset_index() # address the ISSUE
    voting_results = voting_table.loc[voting_table['meeting']== meeting_list[i]].copy()
    voting_results['name'] = voting_results['name'].str.capitalize()
    labeled_data_temp = pd.merge(voting_results, data, how='left', on='name', sort=False)
    labeled_data = pd.concat([labeled_data, labeled_data_temp], axis=0, ignore_index=True)

# there are 8 people who never speak but cast the vote in the meeting, delete them
labeled_data = labeled_data.dropna().reset_index(drop=True)
labeled_data.to_feather('labeled_dataset.ftr')

############################################################################################
# Part II, prepare the parsed labeled data set
import pandas as pd
from functools import reduce
import re
import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_md")

# (a) generate the econ-related phrases
# Oxford Dictionary of Economics: https://www.oxfordreference.com/display/10.1093/acref/9780198759430.001.0001/acref-9780198759430
# Oxford Dictionary of Finance and Banking: https://www.oxfordreference.com/display/10.1093/acref/9780198789741.001.0001/acref-9780198789741
term1 = pd.read_feather('oxford_dic_econ.ftr')
term2 = pd.read_feather('oxford_dic_fb.ftr')

term1 = term1.loc[1:,]  # remove the term 1992
term1_cap = term1['terms'].str.capitalize()
term1 = term1['terms'].to_list() + term1_cap.to_list()

term2_cap = term2['terms'].str.capitalize()
term2 = term2['terms'].to_list() + term2_cap.to_list()

phrases = term1 + term2
phrases = list(set(phrases)) # drop duplicated items

# drop some common words, like or, will
phrases = [x for x in phrases if x !='or' and x !='will' and x !='Or' and x != 'Will' and x != 'OR' and x !='WILL']

# (b) only keep sentences that contain above phrases
data = pd.read_feather('labeled_dataset.ftr')

# only keep related sentences
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(text) for text in phrases]
phrase_matcher.add('econ', patterns)

for i in range(data.shape[0]):
    doc = nlp(data.loc[i, 'words'])
    # extract the related sentences
    temp = [sent.text for sent in doc.sents if phrase_matcher(nlp(sent.text)) != []]
    # check the # of words in each sentence, then delete the ones that contain less than 5 words.
    temp = pd.DataFrame({'words': temp})
    temp['length'] = temp['words'].apply(lambda x: len(x.split()))
    temp = temp.loc[temp['length']>5]
    # combine a list of string into one string
    data.loc[i, 'parsed_words'] = ''.join(temp['words'])  
    if i % 100 == 0:
        print (i, 'is done.\n')


# check the number of sentences each person said
data['num_of_sents'] = data['parsed_words'].apply(lambda x : len([sent.text.strip() for sent in nlp(x).sents if sent.text.strip()!= '']))

data.to_feather('parsed_labeled_dataset_till2016.ftr')

############################################################################################
# Part III, generate the title info for each meeting

voting_table = pd.read_csv('voting_results.csv')
meeting_list = voting_table['meeting'].drop_duplicates().reset_index(drop=True)

# 1: CHAIR 2: VICE CHAIR
title_data = pd.DataFrame()
for i in range(meeting_list.shape[0]):

    title_data_temp = pd.read_feather(file_path+meeting_list[i]+'.ftr')
    title_data_temp['title'] = 0
    title_data_temp.loc[title_data_temp['name'].str.contains("^CHAIR"), 'title'] = 1
    title_data_temp.loc[title_data_temp['name'].str.contains("VICE CHAIR"), 'title'] = 2
    title_data_temp = title_data_temp.loc[title_data_temp['title']>0, ['name', 'title']].copy()
    title_data_temp['name'] = title_data_temp['name'].str.replace(r'(CHAIR|VICE)', '', regex=True)
    title_data_temp['name'] = title_data_temp['name'].str.strip()  # remove extra empty space
    title_data_temp['name'] = title_data_temp['name'].str.capitalize() # convert the names to be first letter uppercase
    title_data_temp['meeting'] = meeting_list[i]

    title_data = pd.concat([title_data, title_data_temp], axis=0, ignore_index=True)

title_data.to_feather('title_info_till2016.ftr')