# this file process speeches from the Federal Reserve Board. Speeches from presidents can be done in similar way and data needs to be collected from local Federal Reserve Bank websites.

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

############################################################
# (1) extract speech links

# all_url = 'https://www.federalreserve.gov/newsevents/speeches.htm'
# from 2011 to 2022 format:
# https://www.federalreserve.gov/newsevents/speech/2022-speeches.htm
# from 2006 to 2010 format:
# https://www.federalreserve.gov/newsevents/speech/2010speech.htm
# from 1996 to 2005 format:
# https://www.federalreserve.gov/newsevents/speech/2005speech.htm

event_link = []
event_speaker = []
event_title = []
event_date = []
for i in range(2011, 2023):
    page_url = 'https://www.federalreserve.gov/newsevents/speech/'+str(i)+'-speeches.htm'
    soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
    temp = soup.find('div', class_="row eventlist")
    event_list = temp.find_all('div', class_="row")

    for event in event_list:
        event_date.append(event.find('time').get_text())
        event_title.append(event.find('em').get_text())
        event_link.append('https://www.federalreserve.gov' + event.find('em').parent.get('href'))
        event_speaker.append(event.find(class_="news__speaker").get_text())
    
    print(i, 'is done.\n')
    time.sleep(5)

data1 = pd.DataFrame({'title': event_title, 'date': event_date, 'speaker': event_speaker, 'link': event_link})
data1['date'] = pd.to_datetime(data1['date'])

event_link = []
event_speaker = []
event_title = []
event_date = []
for i in range(2006, 2011):
    page_url = 'https://www.federalreserve.gov/newsevents/speech/'+str(i)+'speech.htm'
    soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
    temp = soup.find('div', class_="row eventlist")
    event_list = temp.find_all('div', class_="row")

    for event in event_list:
        event_date.append(event.find('time').get_text())
        event_title.append(event.find('em').get_text())
        event_link.append('https://www.federalreserve.gov' + event.find('em').parent.get('href'))
        event_speaker.append(event.find(class_="news__speaker").get_text())
    
    print(i, 'is done.\n')
    time.sleep(5)

data2 = pd.DataFrame({'title': event_title, 'date': event_date, 'speaker': event_speaker, 'link': event_link})
data2['date'] = pd.to_datetime(data2['date'])

# we can combine data1 and data2 together, but the links in data3 is different, so leave data3 alone
data1 = pd.concat([data1, data2], axis=0, ignore_index=True)
data1.to_feather('speech_list_since_2006.ftr')

event_link = []
event_speaker = []
event_title = []
event_date = []
for i in range(1996, 2006):
    page_url = 'https://www.federalreserve.gov/newsevents/speech/'+str(i)+'speech.htm'
    soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
    temp = soup.find('ul', id="speechIndex")
    event_list = temp.find_all('li')

    for event in event_list:
        event_data = [item.strip() for item in event.get_text().split('\n') if item.strip() != '']
        event_date.append(event_data[0])
        event_title.append(event.find('div', class_='speaker').get_text())
        event_link.append('https://www.federalreserve.gov' + event.find('a').get('href'))
        event_speaker.append(event_data[2])

    print(i, 'is done.\n')
    time.sleep(5)

data3 = pd.DataFrame({'title': event_title, 'date': event_date, 'speaker': event_speaker, 'link': event_link})
# in 1997 the title and author info are flipped
data3['date']= pd.to_datetime(data3['date'])
data1997 = data3.loc[(data3['date'] >= '1997-01-01') &  (data3['date'] <= '1997-12-03')]
data3.loc[(data3['date'] >= '1997-01-01') &  (data3['date'] <= '1997-12-03'), 'title'] = data1997.loc[:, 'speaker']
data3.loc[(data3['date'] >= '1997-01-01') &  (data3['date'] <= '1997-12-03'), 'speaker'] = data1997.loc[:, 'title']

data3.to_feather('speech_list_before_2006.ftr')

###################################################################################################
###################################################################################################

# (2) extract speeches

# (a) after 2016
speech_list = pd.read_feather('speech_list_since_2006.ftr')
speech_data = speech_list.copy()
speech_data['speech'] = ''

for i in range(speech_data.shape[0]):
    try:
        page_url = speech_data.loc[i, 'link']
        soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
        temp = soup.find('div', id="article").find('div', class_="col-xs-12 col-sm-8 col-md-8").find_all('p')
        text = ''
        for item in temp:
            new_text = item.get_text().strip()
            text = text + ' ' + new_text
        speech_data.loc[i, 'speech'] = text
    except:
        speech_data.loc[i, 'speech'] = 'Something went wrong...'
    
    time.sleep(10)
    if i % 50 == 0:
        print(i, 'is done.\n')

speech_data.to_feather('speech_data_since_2006.ftr')

# (b) before 2016
speech_list = pd.read_feather('speech_list_before_2006.ftr')
speech_data = speech_list.copy()
speech_data['speech'] = ''

for i in range(speech_data.shape[0]):
    try:
        page_url = speech_data.loc[i, 'link']
        soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
        temp = soup.find_all('td')
        text = ''
        for item in temp:
            if len(item.find_all('p')) != 0:
                new_text = item.get_text().replace('(\n|\r|\r\n|\n\r)', '').strip()
                text = text + ' ' +new_text
        speech_data.loc[i, 'speech'] = text
    except:
        speech_data.loc[i, 'speech'] = 'Something went wrong...'
    
    time.sleep(10)
    if i % 50 == 0:
        print(i, 'is done.\n')

speech_data.to_feather('speech_data_before_2006.ftr')


###################################################################################################
###################################################################################################

# refine the speech data
data1 = pd.read_feather('speech_data_since_2006.ftr')
data2 = pd.read_feather('speech_data_before_2006.ftr')

# after 2016 part
# index 724, Will Monetary Policy Become More of a Science? , 2007-09-21, this is a paper, delete
# index 729, Housing and the Monetary Transmission Mechanism, 2007-09-01, this is a paper, delete
# index 625, Comments on "Some Benefits and Risks of a Hot Economy", 2022-02-18, this one has no content, delete
data1=data1.drop(index=[724, 729, 625]) 
data1['speech'] = data1['speech'].str.replace('\r', ' ', regex=False)
data1['speech'] = data1['speech'].str.replace('\n', ' ', regex=False)
data1['speech'] = data1['speech'].str.replace('\t', ' ', regex=False)
data1['speech'] = data1['speech'].str.replace('\xa0', ' ', regex=False)
data1['speech'] = data1['speech'].str.replace('\xad', ' ', regex=False)


# before 2016 part
# index: 383, Economic development and financial literacy, 2002-01-10, the parsing label is different
page_url = data2.loc[383, 'link']
soup=BeautifulSoup(getHTMLText(page_url), 'html.parser')
temp = soup.find_all('td')
text = ''
for item in temp:
    if len(item.find_all('br')) != 0:
        new_text = item.get_text().replace('(\n|\r|\r\n|\n\r)', '').strip()
        text = text + ' ' +new_text
data2.loc[383, 'speech'] = text

data2['speech'] = data2['speech'].str.replace('\r', ' ', regex=False)
data2['speech'] = data2['speech'].str.replace('\n', ' ', regex=False)
data2['speech'] = data2['speech'].str.replace('\t', ' ', regex=False)
data2['speech'] = data2['speech'].str.replace('\xa0', ' ', regex=False)
data2['speech'] = data2['speech'].str.replace('\xad', ' ', regex=False)

data2['speech'] = data2['speech'].apply(lambda x : re.sub('\s{0,3}Remarks.{1,400}\s\s+\w{3,9}\s\d{1,2},\s\d{4}', '', x))
data2['speech'] = data2.apply(lambda x: re.sub(x['title'], '', x['speech']), axis=1) # remove title from the speech
data2.loc[383, 'speech'] = data2.loc[383, 'speech'].replace('Economic Development and Financial Literacy', '').strip() # remove title from the speech

# combine them together
data = pd.concat([data1, data2], axis=0, ignore_index=True).sort_values(by='date')
# remove Footnotes
data['speech'] = data['speech'].apply(lambda x : re.sub('(Footnotes|Footnote).+', '', x)) # remove all footnotes
data['speech'] = data['speech'].apply(lambda x : re.sub('\.\d{1,2}', '\. ', x)) # remove all footnote numbers
# remove References and Appendix
data['speech'] = data['speech'].apply(lambda x : re.sub('(References).+', '', x))
data['speech'] = data['speech'].apply(lambda x : re.sub('(Appendix|Appendixes|Appendices).+', '', x))

# refine the speaker name section
data.loc[829, 'speaker'] = 'Chairman Ben S. Bernanke'
data.loc[899, 'speaker'] = 'Chairman Ben S. Bernanke'
data.loc[894, 'speaker'] = 'Director, Division of Monetary Affairs, Brian F. Madigan'
data.loc[458, 'speaker'] = 'Governor Ben S. Bernanke'
data.loc[424, 'speaker'] = 'Governor Donald L. Kohn'


data.reset_index(drop=True).to_feather('speech_data/speech_data_board.ftr')


