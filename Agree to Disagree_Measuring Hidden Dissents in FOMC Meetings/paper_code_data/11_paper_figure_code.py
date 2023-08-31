# this file plots all the figures in paper

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

##################################################################
# figure 1: number of vote for and against
voting_pred = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_pred.loc[voting_pred['vote']==2, 'vote'] = 1
voting_pred['date'] = pd.to_datetime(voting_pred['meeting'].str[4:12])
voting_pred['year'] = voting_pred['date'].dt.year
vote_record_data = voting_pred[['meeting', 'name', 'vote', 'year']].copy()

vote_stat = vote_record_data.groupby(['year'], as_index=False).agg({'vote': ['count', 'sum']})
vote_stat.columns = ['year', 'vote_count', 'vote_against']
vote_stat['vote_for'] = vote_stat['vote_count'] - vote_stat['vote_against']
vote_stat['vote_against_prc'] = vote_stat['vote_against'] / vote_stat['vote_count']

# total percentage of NO votes
vote_stat['vote_against'].sum()/vote_stat['vote_count'].sum()

fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.bar(vote_stat['year'], vote_stat['vote_against'],  color=['maroon'], edgecolor='black', alpha=0.7, label='num of NO votes')
ax1.bar(vote_stat['year'], vote_stat['vote_for'], bottom=vote_stat['vote_against'], color='white', edgecolor='black', hatch='/', label='num of YES votes')
ax1.set_ylim(0, 200)
ax1.set_yticks(np.arange(0, 201, 50))
ax1.set_xticks(np.arange(1975, max(vote_stat['year'])+1, 5))
ax1.set_xlabel('Year', fontsize=20)  # need to set xlabel in ax1 or ax2
ax1.set_ylabel('Number of Votes', fontsize=20)
ax2 = ax1.twinx()
ax2.plot(vote_stat['year'], vote_stat['vote_against_prc'], linewidth = 2, linestyle='-', marker='o', color='black', label='percentage of NO votes')
ax2.set_ylim(-1.5, 0.4)
ax2.set_yticks(np.arange(0, 0.4, 0.2))
ax2.set_ylabel('Percentage', fontsize=20)
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
# ncol tells python to arrange labels into num of columns, we have two labels, so if ncol=2 then we can put them horizontally
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)


#######################################################
# figuer 2: number of obs and number of meetings

data = pd.read_feather('parsed_labeled_dataset.ftr')
data['date'] = pd.to_datetime(data['meeting'].str[4:12])
data['year'] = data['date'].dt.year

group_data = data.groupby(['year'], as_index=False).agg({'meeting': ['count', 'nunique']})
group_data.columns = ['year', 'num_obs', 'num_meetings']

fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(group_data['year'], group_data['num_obs'], color='black', linewidth=2, linestyle='-', marker='o', alpha=1, label='num of obervations')
ax1.set_ylabel('Number of Observations', fontsize=20) # set ylabel for different ax
ax1.set_ylim(20, 180)
ax1.set_xticks(range(1975, 2017, 5)) # change the x-axis ticks to 5 year intervals
ax1.set_xlabel('Year', fontsize=20)  # need to set xlabel in ax1 or ax2
ax2 = ax1.twinx()
ax2.bar(group_data['year'], group_data['num_meetings'], color='maroon',  alpha=0.7, label='num of meetings')
ax2.set_ylabel('Number of Meetings', fontsize=20)
ax2.set_ylim(4, 20)
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
# ncol tells python to arrange labels into num of columns, we have two labels, so if ncol=2 then we can put them horizontally
plt.legend(handles, labels, loc='upper right', ncol=2, fontsize=20)

#######################################################
# figure 3: speeches per year
speech_data = pd.read_feather('parsed_speech.ftr')
speech_data = speech_data.loc[(speech_data['date']>='1970-01-01') & (speech_data['date']<'2023-01-01')]
voting_pred = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')

speech_data['year'] = speech_data['date'].dt.year
speech_data['month'] = speech_data['date'].dt.month
voting_pred['date'] = pd.to_datetime(voting_pred['meeting'].str[4:12])
voting_pred['year'] = voting_pred['date'].dt.year
voting_pred['month'] = voting_pred['date'].dt.month

speech_stat_data = pd.DataFrame(speech_data['year'].value_counts()).reset_index()
speech_stat_data.columns = ['year', 'num_speech']
speech_stat_data['year'] = speech_stat_data['year'].astype(int)
speech_stat_data.sort_values(by=['year'], inplace=True)

voting_stat_data = pd.DataFrame(voting_pred['year'].value_counts()).reset_index()
voting_stat_data.columns = ['year', 'num_voting']
voting_stat_data['year'] = voting_stat_data['year'].astype(int)
voting_stat_data.sort_values(by=['year'], inplace=True)

plt.rc('figure', figsize=(20, 10), dpi =100)
plt.plot(voting_stat_data['year'], voting_stat_data['num_voting'], color='black', linewidth=2, linestyle='-', marker='o', alpha=1, label='num of observations')
plt.bar(speech_stat_data['year'], speech_stat_data['num_speech'], color='maroon', alpha=0.7, label='num of speeches')
plt.xticks(np.arange(1970, max(speech_stat_data['year'])+1, 5))
plt.xlabel('Year', fontsize=20)
plt.ylabel('Number', fontsize=20)
plt.legend(fontsize=20)

#######################################################
# figure 5: plot out prediction scores and voting results
# Create and show the box plot
# The box extends from the Q1 to Q3 quartile values of the data, with a line at the median (Q2). The whiskers extend from the edges of box to show the range of the data. By default, they extend no more than 1.5 * IQR (IQR = Q3 - Q1) from the edges of the box, ending at the farthest data point within that interval. Outliers are plotted as separate dots.

data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
data['date'] = pd.to_datetime(data['meeting'].str[4:12])
data['year'] = data['date'].dt.year
data.loc[data['vote']==2, 'vote'] = 1
group_data = data.groupby(['year'], as_index=False).agg({'vote': ['sum', 'count']})
group_data.columns = ['year', 'num_against', 'num_vote']
#group_data['against_prct'] = group_data['num_against'] / group_data['num_vote'] 
group_data['against_prct'] = group_data['num_against'] / data.vote.sum()

fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
# convert year column to string
data['year'] = data['year'].astype(str)
group_data['year'] = group_data['year'].astype(str)
# plot boxplot on first y-axis
sns.boxplot(x='year', y='pred', data=data, color = 'maroon', boxprops=dict(alpha=0.7), showfliers=False, ax=ax1) # showfliers=False to hide outliers
# create second y-axis
ax2 = ax1.twinx()
# plot line plot on second y-axis
sns.lineplot(x='year', y='against_prct', data=group_data, linewidth=2, linestyle='-', marker='o', color='black', ax=ax2)
# set labels and limits
ax1.set_ylabel('Disagreement Score', fontsize=20)
ax1.set_xlabel('Year', fontsize=20)
ax2.set_ylabel('Percentage of NO Votes', fontsize=20)
ax2.set_ylim(-0.1, 0.1)
ax2.set_yticks(np.arange(0, 0.1, 0.02))
plt.show()


##########################################################
# figure 6  word cloud

data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
data['date'] = pd.to_datetime(data['meeting'].str[4:12])

fomc_action = pd.read_csv('FOMC_policy_actions.csv')
fomc_action['rate_increase'] = 0
fomc_action['rate_decrease'] = 0
fomc_action.loc[fomc_action['policy_change']==1, 'rate_increase'] = 1
fomc_action.loc[fomc_action['policy_change']==-1, 'rate_decrease'] = 1
fomc_action['date'] = pd.to_datetime(fomc_action['date'])
fomc_action = fomc_action[['date', 'rate_increase', 'rate_decrease']]

data = pd.merge(data, fomc_action, how='left', on='date')

# Define a custom function that returns a gray color based on frequency
def grey_color_func(word, **kwargs):
    freq = wc.words_[word]
    return "hsl(0, 0%%, %d%%)" % (100 - int(freq * 80))

# Create a list of words
data0 = data.loc[(data['pred']<0.5)&(data['rate_increase']==0)&(data['rate_decrease']==1)]
data1 = data.loc[(data['pred']>=0.5)&(data['rate_increase']==0)&(data['rate_decrease']==1)]
text0 = " ".join(item for item in data0['parsed_words'])
text1 = " ".join(item for item in data1['parsed_words'])
text0 = re.sub('don.{0,3}t', '', text0)
text1 = re.sub('don.{0,3}t', '', text1)
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(['going', 'may', 'think', 'say', 'percent', 'year', 'much', 'look', 'seem', 's', 'S', 'one', 'now', 'will', 'time', 'u', 'see', 'well', 'seems', 'way', 'point', 'lot', 'go', 'might', 'even', 'make', 'number', 'range', 'want', 'rate', 'still', 'question'])

# Create and generate a word cloud image
wordcloud0 = WordCloud(stopwords=stopwords, background_color="white", max_words=200, width=3200, height=1600).generate(text0)
wordcloud1 = WordCloud(stopwords=stopwords, background_color="white", max_words=200, width=3200, height=1600).generate(text1)

# Display the image
wc = wordcloud0
plt.figure(figsize=(32,16), facecolor='white')
plt.imshow(wordcloud0.recolor(color_func=grey_color_func), interpolation="bilinear")
plt.axis("off")
plt.show()

wc = wordcloud1
plt.figure(figsize=(32,16), facecolor='white')
plt.imshow(wordcloud1.recolor(color_func=grey_color_func), interpolation='bilinear')
plt.axis("off")
plt.show()

# for vote records
data3 = data.loc[(data['vote']==0)&(data['rate_increase']==1)&(data['rate_decrease']==0)]
data4 = data.loc[(data['vote']!=0)&(data['rate_increase']==1)&(data['rate_decrease']==0)]
text3 = " ".join(item for item in data3['parsed_words'])
text4 = " ".join(item for item in data4['parsed_words'])
text3 = re.sub('don.{0,3}t', '', text3)
text4 = re.sub('don.{0,3}t', '', text4)
stopwords = set(STOPWORDS)
stopwords.update(['going', 'may', 'think', 'say', 'percent', 'year', 'much', 'look', 'seem', 's', 'S', 'one', 'now', 'will', 'time', 'u', 'see', 'well', 'seems', 'way', 'point', 'lot', 'go', 'might', 'even', 'make', 'number', 'range', 'want', 'rate', 'still', 'question'])
wordcloud3 = WordCloud(stopwords=stopwords, background_color="white", max_words=200, width=3200, height=1600).generate(text3)
wordcloud4 = WordCloud(stopwords=stopwords, background_color="white", max_words=200, width=3200, height=1600).generate(text4)

wc = wordcloud3
plt.figure(figsize=(32,16), facecolor='white')
plt.imshow(wordcloud3.recolor(color_func=grey_color_func), interpolation='bilinear')
plt.axis("off")
plt.show()

wc = wordcloud4
plt.figure(figsize=(32,16), facecolor='white')
plt.imshow(wordcloud4.recolor(color_func=grey_color_func), interpolation='bilinear')
plt.axis("off")
plt.show()


#############################################################
# figure 7 voting dissent and speech dissent

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

speech_data = pd.read_feather('parsed_speech.ftr')
speech_chunks_pred = pd.read_feather('speech_chunks_with_pred.ftr')
voting_pred = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')

voting_pred.loc[voting_pred['vote']==2, 'vote'] = 1

# prepare speech data
# we will see the group_data have fewer obs than speech_data this is because some speeches do not contian any econ-related sentences
group_data=speech_chunks_pred.groupby(['link'], as_index=False).agg({'pred': ['mean']})
group_data.columns=['link', 'pred']
speech_data = pd.merge(speech_data, group_data, on=['link'], how='left', sort=False)
speech_data['year'] = speech_data['date'].dt.year
speech_data['month'] = speech_data['date'].dt.month

# only keep last name
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr.', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].apply(lambda x: x.strip().split(' ')[-1])
speech_data['speaker'] = speech_data['speaker'].str.capitalize()  # only keep the first character capitalized
 
# prepare transcript data
voting_pred['date'] = pd.to_datetime(voting_pred['meeting'].str[4:12])
voting_pred['year'] = voting_pred['date'].dt.year
voting_pred['month'] = voting_pred['date'].dt.month

# link speeches to meeting time, speeches delivered before a certain meeting should be linked to that meeting

meeting_member_info = voting_pred[['meeting', 'date', 'name']].copy()
meeting_info = voting_pred[['meeting', 'date']].copy()
meeting_info = meeting_info.drop_duplicates()
meeting_info = meeting_info.sort_values(by=['date']).reset_index(drop=True)


speech_data['related_meeting'] = np.nan
for i in range(1, meeting_info.shape[0]):
    speech_data.loc[(speech_data['date'] > meeting_info.loc[i-1, 'date']) & (speech_data['date'] <= meeting_info.loc[i, 'date']), 'related_meeting'] = meeting_info.loc[i, 'meeting']

speech_data = speech_data.dropna()
speech_pred = speech_data[['pred', 'related_meeting', 'speaker']].copy()
speech_pred = speech_pred.drop_duplicates()
speech_pred.columns = ['speech_pred', 'meeting', 'name']

voting_pred.rename(columns={'pred': 'voting_pred'}, inplace=True)

speech_score_data = speech_pred[['meeting', 'name', 'speech_pred']].copy()
speech_score_data = speech_score_data.drop_duplicates()
speech_score_data['date'] = pd.to_datetime(speech_score_data['meeting'].str[4:12])
speech_score_data['year'] = speech_score_data['date'].dt.year
speech_score_data['month'] = speech_score_data['date'].dt.month

voting_score_data = voting_pred[['meeting', 'name', 'voting_pred', 'date', 'year', 'month']].copy()
vote_record_data = voting_pred[['meeting', 'name', 'vote', 'date', 'year', 'month']].copy()

###################################################################################
# yearly level results
speech_group_data = speech_score_data[['speech_pred', 'year']].groupby(['year'], as_index=False).agg({'speech_pred': ['mean', 'std']})
speech_group_data.columns = ['year', 'speech_mean', 'speech_std']
speech_group_data['speech_std_mean'] = speech_group_data['speech_std']/speech_group_data['speech_mean']
speech_group_data = speech_group_data.loc[(speech_group_data['year'] >= 1988) & (speech_group_data['year'] <= 2017)]  # only keep data from 1987 to 2017

voting_group_data = voting_score_data[['voting_pred', 'year']].groupby(['year'], as_index=False).agg({'voting_pred': ['mean', 'std']})
voting_group_data.columns = ['year', 'voting_mean', 'voting_std']
voting_group_data['voting_std_mean'] = voting_group_data['voting_std']/voting_group_data['voting_mean']
voting_group_data = voting_group_data.loc[(voting_group_data['year'] >= 1988) & (voting_group_data['year'] <= 2017)]  # only keep data from 1987 to 2017

vote_group_data = vote_record_data[['vote', 'year']].groupby(['year'], as_index=False).agg({'vote': ['sum', 'mean', 'std']})
vote_group_data.columns = ['year', 'vote_sum', 'vote_mean', 'vote_std']
vote_group_data['vote_std_mean'] = vote_group_data['vote_std']/vote_group_data['vote_mean']
vote_group_data = vote_group_data.loc[(vote_group_data['year'] >= 1988) & (vote_group_data['year'] <= 2017)]  # only keep data from 1987 to 2017

# plot for std
fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(speech_group_data['year'], speech_group_data['speech_std'], label='level of disagreement (speech)', linestyle='-', marker='o', color='maroon') # set ylabel for different ax
ax1.plot(voting_group_data['year'], voting_group_data['voting_std'], label='level of disagreement (transcript)', linestyle='-', color='black')
ax1.set_ylim(-0.1, 0.5)
ax1.set_yticks(np.arange(0, 0.6, 0.1))
ax1.set_xlabel('Year', fontsize=20)  # need to set xlabel in ax1 or ax2
ax1.set_ylabel('Level of Disagreement (S.D.)', fontsize=20)
ax2 = ax1.twinx()
ax2.bar(vote_group_data['year'], vote_group_data['vote_sum'], color='maroon',  alpha=0.7, label='num of NO votes')
ax2.set_ylabel('Number of NO Votes', fontsize=20)
ax2.set_ylim(0, 20)
ax2.set_yticks(np.arange(0, 21, 5))
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)


# plot for mean
fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(speech_group_data['year'], speech_group_data['speech_mean'], label='level of disagreement (speech)', linestyle='-', marker='o', color='maroon') # set ylabel for different ax
ax1.plot(voting_group_data['year'], voting_group_data['voting_mean'], label='level of disagreement (transcript)', linestyle='-', color='black')
ax1.set_ylim(-0.1, 0.5)
ax1.set_yticks(np.arange(0, 0.6, 0.1))
ax1.set_xlabel('Year', fontsize=20)  # need to set xlabel in ax1 or ax2
ax1.set_ylabel('Level of Disagreement', fontsize=20)
ax2 = ax1.twinx()
ax2.bar(vote_group_data['year'], vote_group_data['vote_sum'], color='maroon',  alpha=0.7, label='num of NO votes')
ax2.set_ylabel('Number of NO Votes', fontsize=20)
ax2.set_ylim(0, 20)
ax2.set_yticks(np.arange(0, 21, 5))
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)



###################################################################################
# meeting level results
speech_group_data = speech_score_data[['speech_pred', 'date']].groupby(['date'], as_index=False).agg({'speech_pred': ['mean', 'std']})
speech_group_data.columns = ['date', 'speech_mean', 'speech_std']
speech_group_data['speech_std_mean'] = speech_group_data['speech_std']/speech_group_data['speech_mean']
speech_group_data = speech_group_data.loc[(speech_group_data['date'] >= '1988-01-01') & (speech_group_data['date'] < '2018-01-01')]  # only keep data from 1988 to 2017

voting_group_data = voting_score_data[['voting_pred', 'date']].groupby(['date'], as_index=False).agg({'voting_pred': ['mean', 'std']})
voting_group_data.columns = ['date', 'voting_mean', 'voting_std']
voting_group_data['voting_std_mean'] = voting_group_data['voting_std']/voting_group_data['voting_mean']
voting_group_data = voting_group_data.loc[(voting_group_data['date'] >= '1988-01-01') & (voting_group_data['date'] < '2018-01-01')]  # only keep data from 1988 to 2017

vote_group_data = vote_record_data[['vote', 'date']].groupby(['date'], as_index=False).agg({'vote': ['sum','mean', 'std']})
vote_group_data.columns = ['date', 'vote_sum', 'vote_mean', 'vote_std']
vote_group_data['vote_std_mean'] = vote_group_data['vote_std']/vote_group_data['vote_mean']
vote_group_data = vote_group_data.loc[(vote_group_data['date'] >= '1988-01-01') & (vote_group_data['date'] < '2018-01-01')]  # only keep data from 1988 to 2017

# plot for std
fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(speech_group_data['date'], speech_group_data['speech_std'], label='level of disagreement (speech)', linestyle='-', marker='o', color='maroon') # set ylabel for different ax
ax1.plot(voting_group_data['date'], voting_group_data['voting_std'], label='level of disagreement (transcript)', linestyle='-', color='black')
ax1.set_ylim(-0.4, 0.8)
ax1.set_yticks(np.arange(0, 0.8, 0.2))
ax1.set_xlabel('Date', fontsize=20)  # need to set xlabel in ax1 or ax2
ax1.set_ylabel('Level of Disagreement (S.D.)', fontsize=20)
ax2 = ax1.twinx()
#ax2.plot(vote_group_data['date'], vote_group_data['vote_sum'], linestyle='-', marker='o', color='navy', label='num of NO votes')
ax2.bar(vote_group_data['date'], vote_group_data['vote_sum'], width=10,color='gray',  alpha=1, label='num of NO votes')
ax2.set_ylabel('Number of NO Votes', fontsize=20)
ax2.set_ylim(0, 10)
ax2.set_yticks(np.arange(0, 6, 2))
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)


# plot for mean
fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(speech_group_data['date'], speech_group_data['speech_mean'], label='level of disagreement (speech)', linestyle='-', marker='o', color='maroon') # set ylabel for different ax
ax1.plot(voting_group_data['date'], voting_group_data['voting_mean'], label='level of disagreement (transcript)', linestyle='-', color='black')
ax1.set_ylim(-0.4, 0.8)
ax1.set_yticks(np.arange(0, 0.8, 0.2))
ax1.set_xlabel('Date', fontsize=20)  # need to set xlabel in ax1 or ax2
ax1.set_ylabel('Level of Disagreement (Avg.)', fontsize=20)
ax2 = ax1.twinx()
#ax2.plot(vote_group_data['date'], vote_group_data['vote_sum'], linestyle='-', marker='o', color='navy', label='num of NO votes')
ax2.bar(vote_group_data['date'], vote_group_data['vote_sum'], width=10,color='gray',  alpha=1, label='num of NO votes')
ax2.set_ylabel('Number of NO Votes', fontsize=20)
ax2.set_ylim(0, 10)
ax2.set_yticks(np.arange(0, 6, 2))
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)

##########################################
# figure 8: FFR and prediction scores

# see # Part 3: the taylor rule part in 19_paper_regression_code.py


