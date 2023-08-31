##############################################################################################
# (1) Digitalize all the files
##############################################################################################
import pandas as pd
import numpy as np
from pypdf import PdfReader
import re
import os

# this files shows how to prepare data for newly released transcripts.
# transcripts are downloaded from https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

# extract all the file names in the folder
root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/Tsang_Yang_Research/FOMC/transcript_original_files/newly_released/'
files = os.listdir(file_path)
# files.remove('.DS_Store')  # remove the hidden file

# set up the regex
regex = r"\b(CHAIR\w*\s{1,4}\w*\.?|VICE\s{1,4}CHAIR\w*\s{1,4}\w*\.?|MR\.?\s{1,4}\w*\.?|MS\.?\s{1,4}\w*\.?|MRS\.?\s{1,4}\w*\.?|PARTICIPANT\(?\??\)?\.?|SEVERAL\(?\??\)?\.?|SPEAKER\(?\??\)?\.?)"  # use these words to locate speakers

for file in files:
    file_name = file[:-4]
    filepath = file_path + file_name + '.pdf'
    
    try:
        #opening method will be rb
        pdffileobj=open(filepath,'rb')
        #create reader variable that will read the pdffileobj
        pdfreader=PdfReader(pdffileobj)  
        #This will store the number of pages of this pdf file
        page_num=len(pdfreader.pages)

        # find the start page index
        # start_page_index is the actual page number - 1
        for page in range(page_num):
            text_content = pdfreader.pages[page].extract_text()
            if 'Transcript' in text_content:
                start_page_index = page
                break

        # extract data 
        names = []
        data = []
        for page in range(start_page_index, page_num):
            
            # extract the raw data
            text_content = pdfreader.pages[page].extract_text()

            # parse the raw data
            text_content = text_content.strip() 
            text_content = re.sub('\\n\d+', ' ', text_content) 
            # remove page header info
            text_content = re.sub('\s{0,3}[A-Z]\w+\s+\d+.{0,14}\d*,\s*\d+\s+of\s+\d+', '',text_content).strip() 
            text_content = re.sub('\s{0,3}(\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}\s?-\s?\d{1,2}|\d{1,2}/\d{1,2}\s?-\s?\d{1,2}/\d{1,2})/\d{2,4}', ' ', text_content).strip()  
            text_content = re.sub('\s{0,3}\d+\\n', ' ', text_content).strip() 
            text_content = re.sub('-\s?\d{1,3}\s?-', ' ', text_content).strip()
            text_content = re.sub('(\n|\xa0|\xad)', ' ', text_content).strip()
            text_content = re.sub('\[(A|a)pplause\.?\]', '', text_content).strip() 
            text_content = re.sub('\[(L|l)aughter\.?\]', '', text_content).strip()
            text_content = re.sub('\[(U|u)nintelligible\.?\]', '...', text_content).strip()
            text_content = re.sub('\[(S|s)tatement\s?--\s?see\s{0,2}(A|a)ppendix\.?\]', '...', text_content).strip()
            text_content = re.sub('\[', '', text_content).strip()
            text_content = re.sub('\]', '', text_content).strip()
            text_content = re.sub('Transcript .+\d+.+Session', '', text_content).strip() 
            
            # make the formats consistent 
            text_content = re.sub('C\s{0,2}H\s{0,2}A\s{0,2}I\s{0,2}R\s{0,2}M\s{0,2}A\s{0,2}N', 'CHAIRMAN', text_content).strip()
            text_content = re.sub('\d{0,3}CHAIR\w*\W{0,2}', 'CHAIR ', text_content).strip()
            text_content = re.sub('V\s{0,2}I\s{0,2}C\s{0,2}E', 'VICE', text_content).strip()
            text_content = re.sub('\d{0,3}VICE\W{0,2}\s{1,4}CHAIR\w*\W{0,2}', 'VICE CHAIR ', text_content).strip()
            text_content = re.sub('\d{0,3}MR\W{0,2}\.?\s', 'MR. ', text_content).strip()
            text_content = re.sub('\d{0,3}MS\W{0,2}\.?\s', 'MS. ', text_content).strip()
            text_content = re.sub('\d{0,3}MRS\W{0,2}\.?\s', 'MRS. ', text_content).strip()
            text_content = re.sub('\d{0,3}PARTICIPANTS?\W{0,2}\(?\??\)?\.?', 'PARTICIPANT. ', text_content).strip()
            text_content = re.sub('\d{0,3}SEVERAL\W{0,2}\(?\??\)?\.?', 'SEVERAL. ', text_content).strip()
            text_content = re.sub('\d{0,3}SPEAKERS?\W{0,2}\(?\??\)?\.?', 'SPEAKER. ', text_content).strip()
            
            # correct spelling mistakes for YELLEN, DUDLEY
            text_content = re.sub('Y\s{0,2}E\s{0,2}L\s{0,2}L\s{0,2}E\s{0,2}N', 'YELLEN', text_content).strip()
            text_content = re.sub('D\s{0,2}U\s{0,2}D\s{0,2}L\s{0,2}E\s{0,2}Y', 'DUDLEY', text_content).strip()
            
            # change DE POOTER to POOTER, LÓPEZ-SALIDO/LOPEZ-SALIDO to SALIDO
            text_content = re.sub('DE POOTER', 'POOTER', text_content).strip()
            text_content = re.sub('LÓPEZ-SALIDO|LOPEZ-SALIDO', 'SALIDO', text_content).strip()

            names_temp = re.findall(regex, text_content)
            data_temp = [item.strip() for item in re.split(regex, text_content) if item !='\r\n' and item !='\n' and item !='']

            # combine data
            names = names + names_temp  

            if ((data != []) & (data_temp[0] not in names_temp)):
                data[-1] = data[-1] + ' ' + data_temp[0]
                del data_temp[0]
            
            data = data + data_temp
            

        # convert to a dataframe
        if data[0] not in names:
            del data[0]   # remove all the other content like secretary note, etc
            
        final_data = pd.DataFrame({'name': data[::2], 'words':data[1::2]})
        # make it consistent, replace VICE CHAIRMAN , CHAIRWOMAN etc. to VICE CHAIR, CHAIR
        final_data = final_data.replace(regex=r'CHAIR\w*\s', value='CHAIR ')
        final_data = final_data.replace(regex=r'VICE CHAIR\w*\s', value='VICE CHAIR ')
        # clean data
        final_data['name']= final_data['name'].replace(regex=r'\.$', value ='')
        final_data['words']= final_data['words'].replace(regex=r'^\d{1,2}', value = '')
        final_data = final_data.groupby(['name'])['words'].apply(' '.join).reset_index()
        final_data['words']= final_data['words'].replace(regex=r'^\.', value = '')

        # save parsed data
        feather_path = 'parsed_raw_data/'+file_name+'.ftr'   
        final_data.to_feather(feather_path)
    
    except:
        print('ERROR:', file_name)



###############################################################################################
# (2) extract voting results
###############################################################################################
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time

def getHTMLText(url):
    try:
        header={'user-agent':'Mozilla/5.0'}
        r=requests.get(url, timeout=30, headers=header)
        r.raise_for_status() #if the status is not 200, trigger HTTPError
        r.encoding='utf-8'
        return r.text
    except:
        return 'Something went wrong...'

# Step 1: extract minutes url
urls = []
page_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical2017.htm'
soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
data = soup.find_all('a', text='HTML')
urls = ['https://www.federalreserve.gov' + element.get('href') for element in data]
# remove beigebook urls
urls = [element for element in urls if 'beigebook' not in element]

url_data = pd.DataFrame({'url': urls})
url_data['meeting'] = url_data['url'].str.extract('(\d{8})')
url_data['meeting'] = 'FOMC' + url_data['meeting'] + 'meeting'


# Step 2: extract names from minutes
vote_for = []
vote_against = []
for i in range(url_data.shape[0]):
    page_url = url_data.loc[i, 'url']
    try:
        # remember to load getHTMLText() first
        soup=BeautifulSoup(getHTMLText(page_url), 'html.parser') 
        data = soup.find_all(['strong'], string=re.compile('Voting'))
        data = [re.sub('\r?\n?', '', element.next_sibling).strip() for element in data[:2]] 
        vote_for.append(data[0])
        vote_against.append(data[1])
    except:
        vote_for.append('9999')
        vote_against.append('9999')
    
    time.sleep(10)

    print(i, 'is done.\n')

names_data = url_data.copy()
names_data['vote_for'] = vote_for
names_data['vote_against'] = vote_against


# Step 3: refromat the names and check the number of names is correct or not
# extract the right numbers from FOMC_Dissents_Data
fomc_dissents = pd.read_csv('FOMC_Dissents_Data.csv')
fomc_dissents['date'] = fomc_dissents['FOMC Meeting'].str[:-2]+fomc_dissents['Year'].astype(str)
fomc_dissents['date'] = pd.to_datetime(fomc_dissents['date'])
fomc_dissents['meeting_name'] = fomc_dissents['date'].apply(lambda x: x.strftime("%Y%m%d"))
fomc_dissents['meeting_name'] = 'FOMC' + fomc_dissents['meeting_name'] + 'meeting'
fomc_dissents = fomc_dissents[['meeting_name', 'date', 'FOMC Votes', 'Votes for Action', 'Votes Against Action', 'Dissenters Tighter', 'Dissenters Easier', 'Dissenters Other/Indeterminate']]
fomc_dissents.columns = ['meeting', 'date', 'correct_name_num', 'vote_for_num', 'vote_against_num', 'dis_tighter', 'dis_easier', 'dis_other']
fomc_dissents = fomc_dissents.loc[(fomc_dissents['date']>'2016-12-31') & (fomc_dissents['date']< '2018-01-01')]

names_data['vote_for'] = names_data['vote_for'].str.replace(r'\band\b', ',', regex=True)
names_data['vote_for'] = names_data['vote_for'].str.replace(r',\s?,', ',', regex=True)
names_data['vote_for'] = names_data['vote_for'].apply(lambda x: x.split(','))
names_data['vote_for'] = names_data['vote_for'].apply(lambda x: [element.split()[-1] for element in x])
names_data['vote_for'] = names_data['vote_for'].apply(lambda x: ' '.join(x))
names_data['vote_for'] = names_data['vote_for'].str.replace(r'\.', '', regex=True)
names_data['vote_against'] = names_data['vote_against'].str.replace(r'\band\b', ',', regex=True)
names_data['vote_against'] = names_data['vote_against'].str.replace(r',\s?,', ',', regex=True)
names_data['vote_against'] = names_data['vote_against'].apply(lambda x: x.split(','))
names_data['vote_against'] = names_data['vote_against'].apply(lambda x: [element.split()[-1] for element in x])
names_data['vote_against'] = names_data['vote_against'].apply(lambda x: ' '.join(x))
names_data['vote_against'] = names_data['vote_against'].str.replace(r'(None|\.)', '', regex=True)
names_data['name_string'] = names_data['vote_for'] + ' ' + names_data['vote_against']
names_data['name_string'] = names_data['name_string'].str.strip()
names_data['name_num'] = names_data['name_string'].apply(lambda x: len(x.split()))
names_data = pd.merge(names_data, fomc_dissents[['meeting', 'correct_name_num']], how='left', on='meeting', sort=False)
names_data['diff'] = names_data['name_num'] - names_data['correct_name_num']
# outliers = names_data.loc[names_data['diff'] != 0].copy()


# Step 4: build the voting table
# vote_for: 0 , vote_tighter: 1, vote_easier: 2, vote_other: 9
meeting = []
name = []
voting_table0 = pd.DataFrame()
for i in range(names_data.shape[0]):
    meeting_temp = names_data.loc[i,'meeting']
    name_temp = names_data.loc[i,'name_string'].split()
    df_temp = pd.DataFrame({'meeting': meeting_temp, 'name': name_temp})
    voting_table0 = pd.concat([voting_table0, df_temp], axis=0, ignore_index=True)

voting_table0['name'] = voting_table0['name'].str.strip()  #remove empty space
voting_table0['vote_for'] = 0

vote_tighter = fomc_dissents[['meeting', 'dis_tighter']].copy()
vote_tighter = vote_tighter.dropna().reset_index(drop=True)
meeting = []
name = []
voting_table1 = pd.DataFrame()
for i in range(vote_tighter.shape[0]):
    meeting_temp = vote_tighter.loc[i,'meeting']
    name_temp = vote_tighter.loc[i,'dis_tighter'].split(',')
    df_temp = pd.DataFrame({'meeting': meeting_temp, 'name': name_temp})
    voting_table1 = pd.concat([voting_table1, df_temp], axis=0, ignore_index=True)
# voting_table1 is empty, then skip the next 2 steps
voting_table1['name'] = voting_table1['name'].str.strip()  
voting_table1['vote_tighter'] = 1


vote_easier = fomc_dissents[['meeting', 'dis_easier']].copy()
vote_easier = vote_easier.dropna().reset_index(drop=True)
meeting = []
name = []
voting_table2 = pd.DataFrame()
for i in range(vote_easier.shape[0]):
    meeting_temp = vote_easier.loc[i,'meeting']
    name_temp = vote_easier.loc[i,'dis_easier'].split(',')
    df_temp = pd.DataFrame({'meeting': meeting_temp, 'name': name_temp})
    voting_table2 = pd.concat([voting_table2, df_temp], axis=0, ignore_index=True)

voting_table2['name'] = voting_table2['name'].str.strip()  #remove empty space
voting_table2['vote_easier'] = 2


vote_other = fomc_dissents[['meeting', 'dis_other']].copy()
vote_other = vote_other.dropna().reset_index(drop=True)
meeting = []
name = []
voting_table3 = pd.DataFrame()
for i in range(vote_other.shape[0]):
    meeting_temp = vote_other.loc[i,'meeting']
    name_temp = vote_other.loc[i,'dis_other'].split(',')
    df_temp = pd.DataFrame({'meeting': meeting_temp, 'name': name_temp})
    voting_table3 = pd.concat([voting_table3, df_temp], axis=0, ignore_index=True)
# voting_table3 is empty, then skip the next 2 steps
voting_table3['name'] = voting_table3['name'].str.strip()  #remove empty space
voting_table3['vote_other'] = 9

from functools import reduce
data_frames = [voting_table0, voting_table2]
voting_table = reduce(lambda  left,right: pd.merge(left,right,on=['meeting', 'name'], how='left'), data_frames).fillna(0)
voting_table['vote'] = voting_table['vote_for'] + voting_table['vote_easier']

# generate the final data set we need
voting_table = voting_table[['meeting', 'name', 'vote']]
voting_table.to_csv('voting_results_2017.csv', index=False)

###############################################################################################
# (3) generate the labeled dataset
###############################################################################################

# Part I, prepare the raw labeled data set 
voting_table = pd.read_csv('voting_results_2017.csv')

root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/A_Research_Projects/FOMC_project/Code/parsed_raw_data/'

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

labeled_data.to_feather('labeled_dataset_2017.ftr')


# Part II, prepare the parsed labeled data set
import pandas as pd
from functools import reduce
import re
import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load("en_core_web_md")

# (a) generate the econ-related phrases
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
labeled_data = pd.read_feather('labeled_dataset_2017.ftr')

# only keep related sentences
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp(text) for text in phrases]
phrase_matcher.add('econ', patterns)

for i in range(labeled_data.shape[0]):
    doc = nlp(labeled_data.loc[i, 'words'])
    # extract the related sentences
    temp = [sent.text for sent in doc.sents if phrase_matcher(nlp(sent.text)) != []]
    temp = pd.DataFrame({'words': temp})
    temp['length'] = temp['words'].apply(lambda x: len(x.split()))
    temp = temp.loc[temp['length']>5]
    # combine a list of string into one string
    labeled_data.loc[i, 'parsed_words'] = ''.join(temp['words'])  
    
    if i % 20 == 0:
        print (i, 'is done.\n')
    

# check the number of sentences each person said
labeled_data['num_of_sents'] = labeled_data['parsed_words'].apply(lambda x : len([sent.text.strip() for sent in nlp(x).sents if sent.text.strip()!= '']))

labeled_data.to_feather('parsed_labeled_dataset_2017.ftr')


###############################################################################################
# (4) generate title info
###############################################################################################
import pandas as pd
import re
import os

root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/A_Research_Projects/FOMC_project/Code/parsed_raw_data/'

voting_table = pd.read_csv('voting_results_2017.csv')
meeting_list = voting_table['meeting'].drop_duplicates().reset_index(drop=True)

# generate the title info for each meeting
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

title_data.to_feather('title_info_2017.ftr')


###############################################################################################
# (5) combine the additional data with early data
###############################################################################################
import pandas as pd

aug_data_old = pd.read_feather('parsed_labeled_dataset_till2016.ftr')
aug_data_new = pd.read_feather('parsed_labeled_dataset_2017.ftr')

title_old = pd.read_feather('title_info_till2016.ftr')
title_new = pd.read_feather('title_info_2017.ftr')

aug_data = pd.concat([aug_data_old, aug_data_new], axis=0, ignore_index=True)
title_info = pd.concat([title_old, title_new], axis=0, ignore_index=True)

aug_data.to_feather('parsed_labeled_dataset.ftr')
title_info.to_feather('title_info.ftr')
