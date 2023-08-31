# This file extracts FOMC memebrs' voting results in every FOMC meeting. The results will be used to label our training set.
import pandas as pd
import numpy as np
import PyPDF2
import re
import os

# Part I: for meeting before 1993, we extract names from transcripts
# extract all the file names in the folder
root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/transcript_original_files/'
files = os.listdir(file_path)
files.remove('.DS_Store')  # remove the hidden file

# sort files, delete conference calls
files = pd.DataFrame({'meeting': files})
files['meeting_date'] = files['meeting'].str[4:12]
files['meeting_type'] = files['meeting'].str[12:19]
files = files.loc[files['meeting_type']=='meeting'].copy()
del files['meeting_type']
files['meeting_date'] = pd.to_datetime(files['meeting_date'])
files = files.sort_values(by='meeting_date').reset_index(drop=True)

# extract names
meeting = []
names = []
page_index = []
for file in files.loc[0:149, 'meeting']:
    file_name = file[:-4]
    filepath = file_path + file

    pdffileobj=open(filepath,'rb')
    pdfreader=PyPDF2.PdfFileReader(pdffileobj)
    page_num=pdfreader.numPages

    try:
        # find the target page index, the target page contains committee members' names
        for page in range(page_num):
            page_content=pdfreader.getPage(page)
            text_content=page_content.extractText()
            if 'Prefatory' not in text_content and 'Prefatorv' not in text_content and ('PRESENT' in text_content or 'Present' in text_content or 'present' in text_content or 'PARTICIPATING' in text_content):
                target_page_index = page
                break

        text_content = re.sub('\n', ' ', text_content).strip()
        namestring=re.findall('(?:PRESENT|Present|following|PARTICIPATING).+?Messrs', text_content)[0]
        namestring = re.sub('(PRESENT|Present|following|PARTICIPATING):?', '', namestring)
        namestring = re.sub('Messrs', '', namestring).strip()
    except:
        target_page_index = 9999
        namestring = ''

    meeting.append(file_name)
    names.append(namestring)
    page_index.append(target_page_index)

names_data1 = pd.DataFrame({'meeting': meeting, 'name_string': names, 'target_page': page_index})


# there are few files cannot be recognized the begining word 'PRESENT' properly, fill in the names manually
names_data1.loc[41, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich'
names_data1.loc[66, 'name_string'] = 'Volcker Solomon Balles Black Ford Horn Martin Partee Rice Teeters Wallich'
names_data1.loc[69, 'name_string'] = 'Volcker Solomon Balles Black Ford Gramley Horn Martin Partee Rice Teeters Wallich'
names_data1.loc[73, 'name_string'] = 'Volcker Solomon Gramley Guffey Keehn Martin Morris Partee Rice Roberts Teeters Wallich'
names_data1.loc[74, 'name_string'] = 'Volcker Solomon Gramley Guffey Keehn Martin Morris Partee Rice Roberts Teeters Wallich'
names_data1.loc[77, 'name_string'] = 'Volcker Solomon Gramley Guffey Keehn Martin Morris Partee Rice Roberts Teeters Wallich'
names_data1.loc[78, 'name_string'] = 'Volcker Solomon Gramley Guffey Keehn Martin Morris Partee Rice Roberts Teeters Wallich'
names_data1.loc[79, 'name_string'] = 'Volcker Solomon Boehne Boykin Corrigan Gramley Horn Martin Partee Rice Teeters Wallich'
names_data1.loc[98, 'name_string'] = 'Volcker Corrigan Angell Guffey Heller Horn Johnson Melzer Morris Rice Seger Wallich'
names_data1.loc[111, 'name_string'] = 'Greenspan Corrigan Angell Black Forrestal Heller Hoskins Johnson Kelley Parry Seger'
names_data1.loc[133, 'name_string'] = 'Greenspan Corrigan Angell Boehne Boykin Hoskins Kelley LaWare Mullins Seger Stern'

names_data1.to_csv('meeting_names_before1993.csv', index=False)

##################################################################################
# Part II, for meeting after 1993-03-23, we extract names from the FOMC website
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
# (1) from 1993 to 2006
urls = []
for i in range(1993,2007):
    page_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical'+str(i)+'.htm'
    soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
    data = soup.find_all('a', text='Minutes')
    urls_temp = ['https://www.federalreserve.gov' + element.get('href') for element in data]
    urls = urls + urls_temp
    time.sleep(10)

# (2) 2007 this year
page_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical2007.htm'
soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
data1 = soup.find_all('a', text='Minutes')
data2 = soup.find_all('a', text='HTML')
urls_temp = ['https://www.federalreserve.gov' + element.get('href') for element in data1]
urls_temp = urls_temp + ['https://www.federalreserve.gov' + element.get('href') for element in data2]
urls = urls + urls_temp

# (3) from 2008 to 2016
for i in range(2008,2017):
    page_url = 'https://www.federalreserve.gov/monetarypolicy/fomchistorical'+str(i)+'.htm'
    soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
    data = soup.find_all('a', text='HTML')
    urls_temp = ['https://www.federalreserve.gov' + element.get('href') for element in data]
    urls = urls + urls_temp
    time.sleep(10)

# (4) deal with irregular urls
# remove beigebook urls
urls = [element for element in urls if 'beigebook' not in element]
# remove redundant url
urls = [re.sub('http://www.federalreserve.gov', '', element) for element in urls]

url_data = pd.DataFrame({'url': urls})
url_data['meeting'] = url_data['url'].str.extract('(\d{8})')
url_data['meeting'] = 'FOMC' + url_data['meeting'] + 'meeting'

# 1994-12-30 conference call url is broken, delete
url_data = url_data.dropna()
# 1994-04-18 conference call url links to 1994-03-22 meeting, delete
url_data = url_data.drop(index= 10)
# 2001-01-03 conference call url links to 2000-12-19 meeting, delete
url_data = url_data.drop(index= 66)
# 2001-04-11, 2001-04-18 conference call url links to 2001-03-20 meeting, delete
url_data = url_data.drop(index= [69,70])
# 2001-09-13, 2001-09-17 conference call url links to 2001-08-21 meeting, delete
url_data = url_data.drop(index= [74,75])
# 2003-03-35, 2003-04-01, 2003-04-08, 2003-04-16 conference call url links to 2003-03-18 meeting, delete
url_data = url_data.drop(index= [89,90,91,92])
# 2003-09-15 meeting url links to 2003-08-12 meeting, delete
url_data = url_data.drop(index= [96])
# 2007-08-10, 2007-08-16 conference call url links to 2007-09-18 meeting, delete
url_data = url_data.drop(index= [129, 130])


# Step 2: extract names from minutes
vote_for = []
vote_against = []
for i in range(url_data.shape[0]):
    page_url = url_data.loc[i, 'url']
    try:
        # remember to load getHTMLText() first
        soup=BeautifulSoup(getHTMLText(page_url), 'html.parser') 
        data = soup.find_all(['strong','b'], string=re.compile('Voting'))
        data = [re.sub('\r?\n?', '', element.next_sibling).strip() for element in data[:2]] 
        vote_for.append(data[0])
        vote_against.append(data[1])
    except:
        vote_for.append('9999')
        vote_against.append('9999')
    
    time.sleep(10)

    if i % 20 == 0:
        print(i, 'is done.\n')
    
names_data2 = url_data.copy()
names_data2['vote_for'] = vote_for
names_data2['vote_against'] = vote_against

# manually fill in outliers
names_data2.loc[0, 'vote_for'] = 'Mr. Greenspan, Mr. Corrigan, Mr. Angell, Mr. Boehne, Mr. Keehn, Mr. Kelley, Mr. LaWare, Mr. Lindsey, Mr. McTeer, Mr. Mullins, Ms. Phillips, Mr. Stern'
names_data2.loc[0, 'vote_against'] = 'None'
names_data2.loc[43, 'vote_for'] = 'Messrs. Greenspan, McDonough, Ferguson, Gramlich, Hoenig, Kelley, Meyer, Ms. Minehan, Mr. Poole, and Ms. Rivlin.'
names_data2.loc[43, 'vote_against'] = 'Mr. Jordan.'
names_data2.loc[48, 'vote_for'] = 'Messrs. Greenspan, McDonough, Boehne, Ferguson, Gramlich, Kelley, McTeer, Meyer, Moskow, Ms. Rivlin, and Mr. Stern.'
names_data2.loc[48, 'vote_against'] = 'None'
names_data2.loc[49, 'vote_for'] = 'Messrs. Greenspan, McDonough, Boehne, Ferguson, Gramlich, Kelley, McTeer, Meyer, Moskow, Ms. Rivlin, and Mr. Stern.'
names_data2.loc[49, 'vote_against'] = 'None'
names_data2.loc[50, 'vote_for'] = 'Messrs. Greenspan, McDonough, Boehne, Ferguson, Gramlich, Kelley, McTeer, Meyer, Moskow, Ms. Rivlin, and Mr. Stern.'
names_data2.loc[50, 'vote_against'] = 'None'
names_data2.loc[51, 'vote_for'] = 'Messrs. Greenspan, McDonough, Boehne, Ferguson, Gramlich, Meyers, Moskow, Kelley, and Stern.'
names_data2.loc[51, 'vote_against'] = 'Mr. McTeer'
names_data2.loc[79, 'vote_for'] = 'Messrs. Greenspan, McDonough, Bernanke, Ms. Bies, Messrs. Ferguson, Gramlich, Jordan, Kohn, McTeer, Olson, Santomero, and Stern.'
names_data2.loc[79, 'vote_against'] = 'None'
names_data2.loc[102, 'vote_for'] = 'Messrs. Greenspan and Geithner, Ms. Bies, Messrs. Ferguson, Fisher, Kohn, Olson, Moskow, Santomero, and Stern.'
names_data2.loc[102, 'vote_against'] = 'None'
names_data2.loc[103, 'vote_for'] = 'Messrs. Greenspan and Geithner, Ms. Bies, Messrs. Ferguson, Fisher, Kohn, Olson, Moskow, Santomero, and Stern.'
names_data2.loc[103, 'vote_against'] = 'None'
names_data2.loc[167, 'vote_for'] = 'Ben Bernanke, William C. Dudley, James Bullard, Charles L. Evans, Esther L. George, Jerome H. Powell, Jeremy C. Stein, Daniel K. Tarullo, and Janet L. Yellen.'
names_data2.loc[167, 'vote_against'] = 'Eric Rosengren'

# In these following meetings, urls are named after the first date of these meetings, however meeting transcripts are named after the second date of these meeting, correct the names

names_data2.loc[24, 'meeting'] = 'FOMC19960131meeting'
names_data2.loc[27, 'meeting'] = 'FOMC19960703meeting'
names_data2.loc[32, 'meeting'] = 'FOMC19970205meeting'
names_data2.loc[35, 'meeting'] = 'FOMC19970702meeting'
names_data2.loc[40, 'meeting'] = 'FOMC19980204meeting'
names_data2.loc[43, 'meeting'] = 'FOMC19980701meeting'
names_data2.loc[48, 'meeting'] = 'FOMC19990203meeting'
names_data2.loc[51, 'meeting'] = 'FOMC19990630meeting'


names_data2.to_csv('meeting_names_since1993.csv', index=False)


##################################################################################
# Part III, parse the name data
import pandas as pd
import numpy as np
import re

names_data1 = pd.read_csv('meeting_names_before1993.csv')
names_data2 = pd.read_csv('meeting_names_since1993.csv')

# check number of names in each meeting is correct or not, and manually corrent the incorrect ones
# extract the right numbers from FOMC_Dissents_Data
fomc_dissents = pd.read_csv('FOMC_Dissents_Data.csv')
fomc_dissents['date'] = fomc_dissents['FOMC Meeting'].str[:-2]+fomc_dissents['Year'].astype(str)
fomc_dissents['date'] = pd.to_datetime(fomc_dissents['date'])
fomc_dissents['meeting_name'] = fomc_dissents['date'].apply(lambda x: x.strftime("%Y%m%d"))
fomc_dissents['meeting_name'] = 'FOMC' + fomc_dissents['meeting_name'] + 'meeting'
fomc_dissents = fomc_dissents[['meeting_name', 'date', 'FOMC Votes', 'Votes for Action', 'Votes Against Action', 'Dissenters Tighter', 'Dissenters Easier', 'Dissenters Other/Indeterminate']]
fomc_dissents.columns = ['meeting', 'date', 'correct_name_num', 'vote_for_num', 'vote_against_num', 'dis_tighter', 'dis_easier', 'dis_other']
fomc_dissents = fomc_dissents.loc[(fomc_dissents['date']>'1976-03-01') & (fomc_dissents['date']< '2017-01-01')]

# (1) correct names_data1
del names_data1['target_page']
names_data1['name_string'] = names_data1['name_string'].str.replace(r'(Mr\.|Ms\.|Mrs\.|Chairman|Vice-?|\d/|,|\.)', '', regex=True)
names_data1['name_string'] = names_data1['name_string'].str.replace(r'\s+', ' ', regex=True)
names_data1['name_string'] = names_data1['name_string'].str.replace(r'Alternate for \w+$', '', regex=True)
names_data1['name_num'] = names_data1['name_string'].apply(lambda x: len(x.split()))
names_data1 = pd.merge(names_data1, fomc_dissents[['meeting', 'correct_name_num']], how='left', on='meeting', sort=False)
names_data1['diff'] = names_data1['name_num'] - names_data1['correct_name_num']
# outliers = names_data1.loc[names_data1['diff'] != 0].copy()
# manually correct these outliers, '#' means the mistake was caused by an alternate member voted for an absent sitting member 
names_data1.loc[1, 'name_string'] = 'Burns Volcker Balles Black Coldwell Gardner Jackson Kimbrel Partee Wallich Winn'
names_data1.loc[8, 'name_string'] = 'Burns Volcker Black Coldwell Gardner Jackson Kimbrel Lilly Partee Wallich Winn Guffey'
names_data1.loc[34, 'name_string'] = 'Miller Volcker Baughman Coldwell Eastburn Partee Teeters Wallich Willes Mayo'
names_data1.loc[35, 'name_string'] = 'Miller Volcker Balles Black Coldwell Kimbrel Mayo Partee Teeters Wallich'
names_data1.loc[36, 'name_string'] = 'Miller Volcker Balles Black Coldwell Kimbrel Mayo Partee Teeters Wallich'
names_data1.loc[37, 'name_string'] = 'Miller Volcker Balles Black Coldwell Kimbrel Mayo Partee Teeters Wallich'
names_data1.loc[38, 'name_string'] = 'Miller Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Teeters Wallich'
names_data1.loc[39, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[40, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[41, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[42, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[43, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[44, 'name_string'] = 'Volcker Balles Black Coldwell Kimbrel Mayo Partee Rice Schultz Teeters Wallich Timlen' #
names_data1.loc[45, 'name_string'] = 'Volcker Guffey Morris Partee Rice Roos Schultz Teeters Wallich Winn Timlen' #
names_data1.loc[48, 'name_string'] = 'Volcker Gramley Morris Partee Rice Roos Schultz Solomon Teeters Wallich Winn Balles' #
names_data1.loc[49, 'name_string'] = 'Volcker Gramley Morris Partee Rice Roos Schultz Solomon Teeters Wallich Winn Balles' #
names_data1.loc[55, 'name_string'] = 'Volcker Solomon Boehne Boykin Corrigan Partee Rice Schultz Teeters Wallich Winn' #
names_data1.loc[56, 'name_string'] = 'Volcker Solomon Boehne Boykin Corrigan Gramley Partee Rice Schultz Teeters Wallich Winn' #
names_data1.loc[58, 'name_string'] = 'Volcker Solomon Boykin Corrigan Gramley Keehn Partee Rice Schultz Teeters Wallich Black' #
names_data1.loc[80, 'name_string'] = 'Volcker Solomon Boehne Boykin Corrigan Gramley Horn Martin Partee Rice Wallich '
names_data1.loc[81, 'name_string'] = 'Volcker Solomon Boehne Boykin Corrigan Gramley Horn Martin Partee Rice Seger Wallich'
names_data1.loc[86, 'name_string'] = 'Volcker Corrigan Balles Boehne Boykin Gramley Horn Martin Partee Rice Seger Wallich' #
names_data1.loc[89, 'name_string'] = 'Volcker Corrigan Balles Black Forrestal Keehn Martin Partee Rice Seger Wallich'
names_data1.loc[90, 'name_string'] = 'Volcker Corrigan Balles Black Forrestal Keehn Martin Partee Rice Seger Wallich'
names_data1.loc[91, 'name_string'] = 'Volcker Corrigan Balles Black Forrestal Keehn Martin Partee Rice Seger Wallich'
names_data1.loc[92, 'name_string'] = 'Volcker Corrigan Balles Black Forrestal Keehn Martin Partee Rice Seger'
names_data1.loc[93, 'name_string'] = 'Volcker Corrigan Black Forrestal Guffey Keehn Martin Partee Rice Seger' #
names_data1.loc[94, 'name_string'] = 'Volcker Corriganl Angell Black Forrestal Johnson Keehn Martin Parry Rice Seger Wallich'
names_data1.loc[96, 'name_string'] = 'Volcker Corrigan Angell Guffey Horn Johnson Melzer Morris Rice Seger Wallich'
names_data1.loc[99, 'name_string'] = 'Volcker Corrigan Angell Guffey Heller Horn Johnson Melzer Morris Rice Seger Wallich'
names_data1.loc[109, 'name_string'] = 'Greenspan Corrigan Angell Boehne Boykin Heller Johnson Keehn Kelley Seger Stern'
# delete the FOMC19760329meeting, which contains no voting results in our transcript and this meeting does not exist in the FOMC_Dissents_Data
names_data1 = names_data1.drop(index=0)
# now check it again
names_data1['name_num'] = names_data1['name_string'].apply(lambda x: len(x.split()))
names_data1['diff'] = names_data1['name_num'] - names_data1['correct_name_num']
# outliers = names_data1.loc[names_data1['diff'] != 0].copy()

# (2) correct names_data2
del names_data2['url']
# since 2010, they begin to record full names instead of surnames, so this data set has to be divided into two parts
data1 = names_data2.loc[:135].copy()
data1['vote_for'] = data1['vote_for'].str.replace(r'(Mr\.|Ms\.|Mrs\.|Messrs\.|Mses\.|\band\b|\d/|:|,|\.)', ' ', regex=True)
data1['vote_for'] = data1['vote_for'].str.replace(r'\s+', ' ', regex=True)
data1['vote_for'] = data1['vote_for'].str.replace(r'\(.+\)', '', regex=True)
data1['vote_against'] = data1['vote_against'].str.replace(r'(Mr\.|Ms\.|Mrs\.|Messrs\.|Mses\.|None|\band\b|\d/|:|,|\.)', ' ', regex=True)
data1['vote_against'] = data1['vote_against'].str.replace(r'\s+', ' ', regex=True)
data1['name_string'] = data1['vote_for'] + data1['vote_against']
data1['name_string'] = data1['name_string'].str.strip()

data2 = names_data2.loc[136:].copy()
data2['vote_for'] = data2['vote_for'].str.replace(r'\band\b', ',', regex=True)
data2['vote_for'] = data2['vote_for'].str.replace(r',\s?,', ',', regex=True)
data2['vote_for'] = data2['vote_for'].apply(lambda x: x.split(','))
data2['vote_for'] = data2['vote_for'].apply(lambda x: [element.split()[-1] for element in x])
data2['vote_for'] = data2['vote_for'].apply(lambda x: ' '.join(x))
data2['vote_for'] = data2['vote_for'].str.replace(r'\.', '', regex=True)
data2['vote_against'] = data2['vote_against'].str.replace(r'\band\b', ',', regex=True)
data2['vote_against'] = data2['vote_against'].str.replace(r',\s?,', ',', regex=True)
data2['vote_against'] = data2['vote_against'].apply(lambda x: x.split(','))
data2['vote_against'] = data2['vote_against'].apply(lambda x: [element.split()[-1] for element in x])
data2['vote_against'] = data2['vote_against'].apply(lambda x: ' '.join(x))
data2['vote_against'] = data2['vote_against'].str.replace(r'(None|\.)', '', regex=True)
data2['name_string'] = data2['vote_for'] + ' ' + data2['vote_against']
data2['name_string'] = data2['name_string'].str.strip()

names_data2 = pd.concat([data1, data2], axis=0, ignore_index=True)
names_data2['name_num'] = names_data2['name_string'].apply(lambda x: len(x.split()))
names_data2 = pd.merge(names_data2, fomc_dissents[['meeting', 'correct_name_num']], how='left', on='meeting', sort=False)
names_data2['diff'] = names_data2['name_num'] - names_data2['correct_name_num']
# outliers = names_data2.loc[names_data2['diff'] != 0].copy()

# (3) combine names_data1 and names_data2 together
names_data = pd.concat([names_data1[['meeting', 'name_string']], names_data2[['meeting', 'name_string']]], axis=0, ignore_index=True)

# the names_data contains fewer meetings compared with the data in FOMC_Dissents_Data, check these outliers
checkdata = pd.merge(fomc_dissents[['meeting', 'correct_name_num']], names_data, how='left', on='meeting', sort=False)
outliers = checkdata.loc[checkdata['name_string'].isna()]

# manually add these 13 meetings info to names_data 
meeting = ['FOMC19780310confcall', 'FOMC19780505confcall', 'FOMC19800307confcall', 'FOMC19800506confcall', 'FOMC19801205confcall',  'FOMC19810224confcall', 'FOMC19810506confcall', 'FOMC19880105confcall', 'FOMC20010103ConfCall', 'FOMC20010418ConfCall', 'FOMC20010917ConfCall', 'FOMC20080121confcall', 'FOMC20081007confcall']
names_string = [
    'Miller Volcker Burns Coldwell Eastburn Jackson Kimbrel Wallich Willes Winn',
    'Miller Volcker Baughman Black Gardner Jackson Partee Wallich Willes Winn',
    'Volcker Guffey Morris Partee Rice Roos Schultz Teeters Timlen Wallich Winn',
    'Volcker Guffey Mayo Morris Rice Roos Schultz Solomon Teeters Wallich',
    'Volcker Solomon Gramley Guffey Morris Partee Rice Roos Teeters Wallich Winn',
    'Volcker Gramley Guffey Morris Partee Rice Roos Schultz Teeters Winn',
    'Volcker Solomon Boehne Boykin Gramley Rice Schultz Teeters Winn',
    'Greenspan Corrigan Angell Boehne Boykin Heller Johnson Keehn Kelley Seger Stern',
    'Greenspan McDonough Ferguson Gramlich Hoenig Kelley Meyer Minehan Moskow Poole',
    'Greenspan McDonough Ferguson Gramlich Hoenig Kelley Meyer Minehan Moskow Poole',
    'Greenspan McDonough Ferguson Gramlich Hoenig Kelley Meyer Minehan Moskow Poole',
    'Bernanke Geithner Evans Hoenig Kohn Kroszner Poole Rosengren Warsh',
    'Bernanke Geithner Duke Fisher Kohn Kroszner Pianalto Plosser Stern Warsh'
]
add_meeting = pd.DataFrame({'meeting': meeting, 'name_string': names_string})
names_data = pd.concat([names_data, add_meeting], axis=0, ignore_index=True)

# change the meeting names in FOMC_Dissents_Data
fomc_dissents = fomc_dissents.reset_index(drop = True)
fomc_dissents.loc[[24, 27, 50, 53, 61, 65, 67, 122, 227, 230, 234, 286, 293], 'meeting'] = ['FOMC19780310confcall', 'FOMC19780505confcall', 'FOMC19800307confcall', 'FOMC19800506confcall', 'FOMC19801205confcall',  'FOMC19810224confcall', 'FOMC19810506confcall', 'FOMC19880105confcall', 'FOMC20010103ConfCall', 'FOMC20010418ConfCall', 'FOMC20010917ConfCall', 'FOMC20080121confcall', 'FOMC20081007confcall']


##################################################################################
# Part IV, build the voting table
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

voting_table3['name'] = voting_table3['name'].str.strip()  #remove empty space
voting_table3['vote_other'] = 9

from functools import reduce
data_frames = [voting_table0, voting_table1, voting_table2, voting_table3]
voting_table = reduce(lambda  left,right: pd.merge(left,right,on=['meeting', 'name'], how='left'), data_frames).fillna(0)
voting_table['vote'] = voting_table['vote_for'] + voting_table['vote_tighter'] + voting_table['vote_easier'] + voting_table['vote_other']

# check if there are mis-match because of name misspelling
voting_table['vote_tighter'].sum() # 174, number equals the row number of voting_table1, check
voting_table['vote_easier'].sum()/2 # 71<73, number is smaller than the row number of voting_table2. This is becuase the FOMC_Dissents_Data recorded that Teeters voted easier in FOMC19801126meeting & FOMC19801212meeting, but we dropped these two meetings because the corresponding transcripts do not contain voting results.
voting_table['vote_other'].sum()/9  # 30, number equals the row number of voting_table3, check

# some names are not read properly by algorithem or mis-spelled on the website, change them
voting_table['name'] = voting_table['name'].str.replace(r'\bCorriganl\b', 'Corrigan', regex = True)
voting_table['name'] = voting_table['name'].str.replace(r'\bMeyers\b', 'Meyer', regex = True)

# generate the final data set we need
voting_table = voting_table[['meeting', 'name', 'vote']]
voting_table.to_csv('voting_results.csv', index=False)