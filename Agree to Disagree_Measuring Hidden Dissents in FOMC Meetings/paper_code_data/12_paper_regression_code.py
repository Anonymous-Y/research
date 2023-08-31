# this file replicates all regression results in paper

import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
from datetime import datetime

########################################################################################################################################
########################################################################################################################################
# Part 1: check the corralation between prediction socres and other variables
########################################################################################################################################
########################################################################################################################################

# FOMC20080121confcall, FOMC20010103ConfCall this two should be removed
# there is no speech data can be linked to FOMC20010103ConfCall, and if speech data is linked to FOMC20080121confcall, then no speech data can be linked to FOMC20080130meeting

# fomc data
fomc_info = pd.read_feather('fomc_info_reg_data.ftr')   
fomc_info = fomc_info.loc[(fomc_info['meeting']!='FOMC20080121confcall') & (fomc_info['meeting']!='FOMC20010103ConfCall')]

# voting data
voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_pred = voting_data[['meeting', 'pred']].copy()
voting_pred = voting_pred.groupby(['meeting'], as_index=False).agg({'pred': ['mean','std']})
voting_pred.columns = ['meeting', 'voting_pred_mean','voting_pred_std']

vote_dist = voting_data[['meeting', 'vote']].copy()
vote_dist.loc[vote_dist['vote']==2, 'vote'] = 1
vote_dist = vote_dist.groupby(['meeting'], as_index=False).agg({'vote': ['mean', 'std']})
vote_dist.columns = ['meeting', 'vote_dist_mean', 'vote_dist_std']

# speech data
# we will see the group_data have fewer obs than speech_data this is because some speeches do not contian any econ-related sentences
speech_data = pd.read_feather('parsed_speech.ftr')
speech_chunks_pred = pd.read_feather('speech_chunks_with_pred.ftr')
group_data=speech_chunks_pred.groupby(['link'], as_index=False).agg({'pred': ['mean']})
group_data.columns=['link', 'pred']
speech_data = pd.merge(speech_data, group_data, on=['link'], how='left', sort=False)
speech_data = speech_data[['title', 'date', 'speaker', 'pred']]
speech_data = speech_data.sort_values(by=['date']).reset_index(drop=True)


# link speeches to meeting time, speeches delivered before a certain meeting should be linked to that meeting
meeting_info = fomc_info[['meeting', 'date']].copy()
meeting_info = meeting_info.drop_duplicates()
meeting_info = meeting_info.sort_values(by=['date']).reset_index(drop=True)


speech_data['related_meeting'] = np.nan
for i in range(1, meeting_info.shape[0]):
    speech_data.loc[(speech_data['date'] >= meeting_info.loc[i-1, 'date']) & (speech_data['date'] < meeting_info.loc[i, 'date']), 'related_meeting'] = meeting_info.loc[i, 'meeting']

speech_data = speech_data.dropna()
speech_pred = speech_data[['pred', 'related_meeting']].copy()
speech_pred = speech_pred.groupby('related_meeting', as_index=False).agg({'pred': ['mean', 'std']})
speech_pred.columns = ['meeting', 'speech_pred_mean', 'speech_pred_std']


# construct reg data
data_frames = [fomc_info, voting_pred, vote_dist, speech_pred]
reg_data_total = reduce(lambda left,right: pd.merge(left,right,on=['meeting'], how='left'), data_frames)


###############################################################
# test unemploy and cpi 

reg_data = reg_data_total.copy()

result1 = smf.ols(formula='voting_pred_mean ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data).fit().get_robustcov_results(cov_type='HC1')

result2 = smf.ols(formula='vote_dist_mean ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data).fit().get_robustcov_results(cov_type='HC1')

result3 = smf.ols(formula='voting_pred_mean ~ experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result4 = smf.ols(formula='vote_dist_mean ~ experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

table = summary_col([result1, result3, result2, result4], stars=True, float_format='%.4f', model_names=['voting_pred_mean', 'voting_pred_mean', 'vote_mean', 'vote_mean'], regressor_order= ['unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std',  'experience', 'age', 'highest_school_wealth', 'gender', 'highest_major', 'hometown_entropy', 'highest_school_entropy', 'app_potus_entropy', 'current_potus_party', 'depression_dis', 'inflation_dis', 'wwii'], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
print(table)

#convert the table to LaTeX
latex_table = table.as_latex()
#print the LaTeX code
print(latex_table)


# result5 = smf.ols(formula='voting_pred_std ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data).fit().get_robustcov_results(cov_type='HC1')

# result6 = smf.ols(formula='vote_dist_std ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data).fit().get_robustcov_results(cov_type='HC1')

# result7 = smf.ols(formula='voting_pred_std ~ experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result8 = smf.ols(formula='vote_dist_std ~ experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')


# table = summary_col([result5, result7, result6, result8], stars=True, float_format='%.4f', model_names=['voting_pred_mean', 'voting_pred_mean', 'vote_mean', 'vote_mean'], regressor_order= ['unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std',  'experience', 'age', 'highest_school_wealth', 'gender', 'highest_major', 'hometown_entropy', 'highest_school_entropy', 'app_potus_entropy', 'current_potus_party', 'depression_dis', 'inflation_dis', 'wwii'], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
# print(table)


# #convert the table to LaTeX
# latex_table = table.as_latex()
# #print the LaTeX code
# print(latex_table)


########################################################################################################################################
########################################################################################################################################
# Part 2: explanation power of speech data
########################################################################################################################################
########################################################################################################################################
# method 1:  voting data only using memebers who give a speech
# fomc data
fomc_info = pd.read_feather('fomc_info_reg_data_only_contain_member_give_speech.ftr')   # only contains FOMC members who gave speeches before meetings
fomc_info = fomc_info.loc[(fomc_info['meeting']!='FOMC20080121confcall') & (fomc_info['meeting']!='FOMC20010103ConfCall')]

# if we only use the members who gave speeches before meetings, then we need to filter out other members' voting data
speech_data = pd.read_feather('parsed_speech.ftr')
speech_chunks_pred = pd.read_feather('speech_chunks_with_pred.ftr')
group_data=speech_chunks_pred.groupby(['link'], as_index=False).agg({'pred': ['mean']})
group_data.columns=['link', 'speech_pred']
speech_data = pd.merge(speech_data, group_data, on=['link'], how='left', sort=False)
speech_data = speech_data[['title', 'date', 'speaker', 'speech_pred']]
speech_data = speech_data.sort_values(by=['date']).reset_index(drop=True)
# only keep last name
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr.', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].apply(lambda x: x.strip().split(' ')[-1])
speech_data['speaker'] = speech_data['speaker'].str.capitalize()  # only keep the first character capitalized
speech_data = speech_data[['date', 'speaker', 'speech_pred']].copy()

voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_data = voting_data[['meeting', 'name','pred', 'vote']].copy()
voting_data.loc[voting_data['vote']==2, 'vote'] = 1
voting_data.rename(columns={'pred': 'voting_pred'}, inplace=True)
voting_data['date'] = pd.to_datetime(voting_data['meeting'].str[4:12])
voting_info = voting_data[['meeting', 'date']].copy()
voting_info = voting_info.drop_duplicates()
voting_info = voting_info.sort_values(by=['date']).reset_index(drop=True)
 
speech_data['related_meeting'] = np.nan
for i in range(1, voting_info.shape[0]):
    speech_data.loc[(speech_data['date'] >= voting_info.loc[i-1, 'date']) & (speech_data['date'] < voting_info.loc[i, 'date']), 'related_meeting'] = voting_info.loc[i, 'meeting']

speech_data = speech_data.dropna()
speech_data = speech_data[['related_meeting', 'speaker', 'speech_pred']].copy()
speech_data = speech_data.drop_duplicates()
# if a member delivered multiple speeches, we calculate the average
speech_data = speech_data.groupby(['related_meeting', 'speaker'], as_index=False).agg({'speech_pred': ['mean']})
speech_data.columns = ['meeting', 'name', 'speech_pred']

# now only keep voting data for members who gave speeches before meetings
reg_data = pd.merge(voting_data, speech_data, on=['meeting', 'name'], how='inner', sort=False)
reg_data = reg_data.groupby(['meeting'], as_index=False).agg({'voting_pred': ['mean', 'std'], 'vote': ['mean', 'std'], 'speech_pred': ['mean', 'std']})
reg_data.columns = ['meeting', 'voting_pred_mean', 'voting_pred_std', 'vote_dist_mean', 'vote_dist_std', 'speech_pred_mean', 'speech_pred_std']
reg_data = pd.merge(fomc_info, reg_data, on=['meeting'], how='left', sort=False) # add the fomc info
reg_data = reg_data.loc[(reg_data['date'] >='1987-09-01') & (reg_data['date'] <'2018-01-01'), :].copy() # Post-Volcker
reg_data = reg_data.dropna()


#####################################################################
# method 2: voting data using full dataset
# fomc data
fomc_info = pd.read_feather('fomc_info_reg_data.ftr')     # all FOMC members
fomc_info = fomc_info.loc[(fomc_info['meeting']!='FOMC20080121confcall') & (fomc_info['meeting']!='FOMC20010103ConfCall')]

# speech data
# we will see the group_data have fewer obs than speech_data this is because some speeches do not contian any econ-related sentences
speech_data = pd.read_feather('parsed_speech.ftr')
speech_chunks_pred = pd.read_feather('speech_chunks_with_pred.ftr')
group_data=speech_chunks_pred.groupby(['link'], as_index=False).agg({'pred': ['mean']})
group_data.columns=['link', 'pred']
speech_data = pd.merge(speech_data, group_data, on=['link'], how='left', sort=False)
speech_data = speech_data[['title', 'date', 'speaker', 'pred']]
speech_data = speech_data.sort_values(by=['date']).reset_index(drop=True)

# link speeches to meeting time, speeches delivered before a certain meeting should be linked to that meeting
meeting_info = fomc_info[['meeting', 'date']].copy()
meeting_info = meeting_info.drop_duplicates()
meeting_info = meeting_info.sort_values(by=['date']).reset_index(drop=True)

speech_data['related_meeting'] = np.nan
for i in range(1, meeting_info.shape[0]):
    speech_data.loc[(speech_data['date'] >= meeting_info.loc[i-1, 'date']) & (speech_data['date'] < meeting_info.loc[i, 'date']), 'related_meeting'] = meeting_info.loc[i, 'meeting']

speech_data = speech_data.dropna()
# only keep last name
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr.', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].apply(lambda x: x.strip().split(' ')[-1])
speech_data['speaker'] = speech_data['speaker'].str.capitalize()  # only keep the first character capitalized
speech_pred = speech_data[['related_meeting', 'speaker', 'pred']].copy()
# if a member delivered multiple speeches, we calculate the average
speech_pred = speech_pred.groupby(['related_meeting', 'speaker'], as_index=False).agg({'pred': ['mean']})
speech_pred.columns = ['related_meeting', 'speaker', 'pred']
speech_pred = speech_pred.groupby('related_meeting', as_index=False).agg({'pred': ['mean', 'std']})
speech_pred.columns = ['meeting', 'speech_pred_mean', 'speech_pred_std']

# voting data
voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_pred = voting_data[['meeting', 'pred']].copy()
voting_pred = voting_pred.groupby(['meeting'], as_index=False).agg({'pred': ['mean','std']})
voting_pred.columns = ['meeting', 'voting_pred_mean','voting_pred_std']

vote_dist = voting_data[['meeting', 'vote']].copy()
vote_dist.loc[vote_dist['vote']==2, 'vote'] = 1
vote_dist = vote_dist.groupby(['meeting'], as_index=False).agg({'vote': ['mean', 'std']})
vote_dist.columns = ['meeting', 'vote_dist_mean', 'vote_dist_std']

# construct reg data
data_frames = [fomc_info, voting_pred, vote_dist, speech_pred]
reg_data = reduce(lambda left,right: pd.merge(left,right,on=['meeting'], how='left'), data_frames)
reg_data = reg_data.loc[(reg_data['date'] >='1987-09-01') & (reg_data['date'] <'2018-01-01'), :].copy() # Post-Volcker
reg_data = reg_data.dropna()

##################################################################################
# test the effect of speech prediction

result1 = smf.ols(formula='voting_pred_mean ~ speech_pred_mean', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result2 = smf.ols(formula='voting_pred_mean ~ speech_pred_mean + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth   + highest_major + app_potus_entropy + current_potus_party  + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result3 = smf.ols(formula='voting_pred_mean ~ speech_pred_mean + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result4 = smf.ols(formula='voting_pred_mean ~ speech_pred_mean + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth  + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result5 = smf.ols(formula='vote_dist_mean ~ speech_pred_mean', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

result6 = smf.ols(formula='vote_dist_mean ~ speech_pred_mean + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth  + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

table = summary_col([result1, result2, result3, result4, result5, result6], stars=True, float_format='%.4f', model_names=['voting_speech', 'voting_speech', 'voting_speech', 'voting_speech', 'vote', 'vote'], regressor_order= ['speech_pred_std', 'speech_pred_mean', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std',  'experience', 'age', 'highest_school_wealth', 'gender', 'highest_major', 'hometown_entropy', 'highest_school_entropy', 'app_potus_entropy', 'current_potus_party', 'depression_dis', 'inflation_dis', 'wwii'], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
print(table)


#convert the table to LaTeX
latex_table = table.as_latex()
#print the LaTeX code
print(latex_table)


# result1 = smf.ols(formula='voting_pred_std ~ speech_pred_std', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result2 = smf.ols(formula='voting_pred_std ~ speech_pred_std + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth   + highest_major + app_potus_entropy + current_potus_party  + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result3 = smf.ols(formula='voting_pred_std ~ speech_pred_std + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result4 = smf.ols(formula='voting_pred_std ~ speech_pred_std + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth  + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result5 = smf.ols(formula='vote_dist_std ~ speech_pred_std', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# result6 = smf.ols(formula='vote_dist_std ~ speech_pred_std + experience + age + gender+ hometown_entropy + highest_school_entropy + highest_school_wealth  + highest_major + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + app_potus_entropy + current_potus_party + wwii + depression_dis + inflation_dis', data = reg_data, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# table = summary_col([result1, result2, result3, result4, result5, result6], stars=True, float_format='%.4f', model_names=['voting_speech', 'voting_speech', 'voting_speech','voting_speech', 'vote', 'vote'], regressor_order= ['speech_pred_std', 'speech_pred_mean', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std',  'experience', 'age', 'highest_school_wealth', 'gender', 'highest_major', 'hometown_entropy', 'highest_school_entropy', 'app_potus_entropy', 'current_potus_party', 'depression_dis', 'inflation_dis', 'wwii'], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
# print(table)


# #convert the table to LaTeX
# latex_table = table.as_latex()
# #print the LaTeX code
# print(latex_table)


########################################################################################################################################
########################################################################################################################################
# Part 3: the taylor rule part
########################################################################################################################################
########################################################################################################################################

# fomc data
fomc_info = pd.read_feather('fomc_info_reg_data.ftr')

######################################################################################################
# Taylor's rule

##### fed rate part
raw_fed_rate = pd.read_csv('FEDFUNDS.csv') # This is intre-bank borrowing rate
raw_fed_rate.columns = ['date', 'fed_rate']
raw_fed_rate['date'] = pd.to_datetime(raw_fed_rate['date'])

# # use shadow rate to replace funds rate after 2008
# shadow_rate = pd.read_csv('shadowrate_US.csv')
# shadow_rate['date'] = shadow_rate['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m'))
# raw_fed_rate = pd.merge(raw_fed_rate, shadow_rate, how='left', on='date')
# raw_fed_rate.loc[raw_fed_rate['date']>=datetime(2008,1,1), 'fed_rate'] = raw_fed_rate.loc[raw_fed_rate['date']>=datetime(2008,1,1), 'shadow_rate']
# raw_fed_rate.drop(columns=['shadow_rate'], inplace=True)

##### fed rate 
fed_rate_quarter = raw_fed_rate.copy()
# pick the middle month for each quarter
# build the lookup table to only extract the middle month in each quarter
middle_month = pd.DataFrame({'month': [2, 5, 8, 11], 'middle_month': 1})
fed_rate_quarter['month'] = fed_rate_quarter['date'].dt.month
fed_rate_quarter = pd.merge(fed_rate_quarter, middle_month, how='left', on='month')
# only keep the middle month
fed_rate_quarter = fed_rate_quarter.loc[fed_rate_quarter['middle_month']==1, :].copy()
fed_rate_quarter.drop(columns=['month', 'middle_month'], inplace=True)
fed_rate_quarter.reset_index(drop=True, inplace=True)

fed_rate_quarter['fed_rate_lag1'] = fed_rate_quarter['fed_rate'].shift(1)
fed_rate_quarter['fed_rate_lag2'] = fed_rate_quarter['fed_rate'].shift(2)


##### gdp gap part
# this is gdp gap, not real gdp
g_gdp_1 = pd.read_csv('Greenbook_Output_Gap_Web_Previous_Format.csv')
g_gdp_1 = g_gdp_1[['Greenbook \nPublication \nDate', 'T.1', 'T+1']]
g_gdp_1.columns = ['GBdate', 'gdp_f0', 'gdp_f1']
g_gdp_1['GBdate'] = pd.to_datetime(g_gdp_1['GBdate'])
g_gdp_1 = g_gdp_1.loc[g_gdp_1['GBdate']<'1996-03-21', :]  # remove the overlap part

g_gdp_2 = pd.read_csv('Greenbook_Output_Gap_DH_Web.csv')
g_gdp_2.rename(columns={'Unnamed: 0': 'quarter'}, inplace=True)
g_gdp_2['quarter'] = g_gdp_2['quarter'].str.replace(':00', '')
g_gdp_2['quarter'] = g_gdp_2['quarter'].str.replace(':0', 'Q')

gdp_date = []
gdp_f_0 = []
gdp_f_1 = []
for i in range(1, len(g_gdp_2.columns)):
    col = g_gdp_2.columns[i]
    temp = col.split('_')[1]
    dt = datetime.strptime(temp, '%y%m%d')
    date = pd.to_datetime(dt)
    quarter = str(pd.to_datetime(dt).to_period('Q'))
    index = g_gdp_2[g_gdp_2['quarter']==quarter].index
    gdp_date.append(date)
    gdp_f_0.append(g_gdp_2.loc[index, col].values[0])
    gdp_f_1.append(g_gdp_2.loc[index+1, col].values[0])

g_gdp = pd.DataFrame({'GBdate': gdp_date, 'gdp_f0': gdp_f_0, 'gdp_f1': gdp_f_1})

# combine two data sets together
g_gdp = pd.concat([g_gdp_1, g_gdp], axis=0, ignore_index=True)


##### cpi part
g_core_cpi = pd.read_csv('core_cpi.csv')
g_core_cpi = g_core_cpi.loc[g_core_cpi['meeting_date'].isna() == False, :]
g_core_cpi = g_core_cpi[['meeting_date', 'GBdate', 'gPCPIXF0', 'gPCPIXF1', 'gPCPIXB1']].copy()
g_core_cpi.rename(columns={'meeting_date': 'date', 'gPCPIXF0': 'cpi_f0', 'gPCPIXF1': 'cpi_f1', 'gPCPIXB1': 'cpi_b1'}, inplace=True)
g_core_cpi['date'] = pd.to_datetime(g_core_cpi['date'])
g_core_cpi['GBdate'] = pd.to_datetime(g_core_cpi['GBdate'], format='%Y%m%d')
# remove all the entries we do not have data
g_core_cpi = g_core_cpi.loc[g_core_cpi['date']>='1986-02-12', :]

# combine gpd and cpi data together
# some GBdata does not match in these two dataset, i correct the dates in the gdp side manually. 
macro_data = pd.merge(g_gdp, g_core_cpi, how='outer', on='GBdate', sort=False)
macro_data.drop(columns='GBdate', inplace=True)

#### voting data should be earlier than fed_rate data
voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_data.loc[voting_data['vote']==2, 'vote']=1
voting_pred = voting_data[['meeting', 'pred', 'vote']].copy()
voting_pred = voting_pred.groupby(['meeting'], as_index=False).agg({'pred': ['mean','std'], 'vote': ['mean', 'std']})
voting_pred.columns = ['meeting', 'pred_mean','pred_std', 'vote_mean', 'vote_std']
voting_pred['date'] = pd.to_datetime(voting_pred['meeting'].str[4:12])
voting_pred = voting_pred[['date', 'pred_mean', 'pred_std', 'vote_mean', 'vote_std']]


# combine all FOMC related data together
cb_df = [voting_pred, macro_data]
fomc_data = reduce(lambda left,right: pd.merge(left,right,on=['date'], how='left'), cb_df)
fomc_data = fomc_data.dropna()
fomc_data.rename(columns={'date': 'meeting_date'}, inplace=True)

# now we match FOMC data timeline with fed_rate data timeline
# fed_rate_quarter will be matched to the nearest meeting/greenbook release that was hold right before the fed_rate released.
reg_data_total = pd.merge_asof(fed_rate_quarter, fomc_data, left_on='date', right_on='meeting_date', direction='backward')
# after 2018-02-01, we do not have voting_pred data avaiable
reg_data_total.loc[reg_data_total['date'] > '2018-02-01', ['meeting_date', 'pred_std', 'pred_mean', 'vote_mean', 'vote_std', 'gdp_f0', 'gdp_f1', 'cpi_f0', 'cpi_f1', 'cpi_b1']] = np.nan
reg_data_total.drop(columns=['meeting_date'], inplace=True)

reg_data_total['dummy_1993']= 0
reg_data_total.loc[reg_data_total['date'] >='1993-01-01', 'dummy_1993']=1  # use 1993 the result is even better.

# reg_data_total.to_feather('taylor_reg_data_origin_rate.ftr')
# reg_data_total.to_feather('taylor_reg_data_shadowrate.ftr')

#############################################
# Volcker: 1979Q3-1987Q2
# Greenspan: 1987Q3-2005Q4 
# Bernanke: 2006Q1-2013Q4
# Post-Volcker 1987Q3-2007Q4
# after 2008 : 2008Q1-2017Q4
#####################################

# Post-Volcker
reg_data2 = reg_data_total.loc[(reg_data_total['date'] >='1987-09-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data2.dropna(inplace=True)
# Before 2008
reg_data3 = reg_data_total.loc[(reg_data_total['date'] >='1987-09-01') & (reg_data_total['date'] <'2008-01-01'), :].copy()
reg_data3.dropna(inplace=True)
# After 2008, wo shadowrate
reg_data4 = reg_data_total.loc[(reg_data_total['date'] >='2008-01-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data4.dropna(inplace=True)
# After 2008, with shadowrate
reg_data5 = pd.read_feather('taylor_reg_data_shadowrate.ftr')
reg_data5 = reg_data5.loc[(reg_data_total['date'] >='2008-01-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data5.dropna(inplace=True)

#################################################
# figure 8 in paper
plot_data = reg_data2
shadow_rate = pd.read_csv('shadowrate_US.csv')
shadow_rate['date'] = shadow_rate['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m'))
plot_data = pd.merge(plot_data, shadow_rate, how='left', on='date')

fig, ax1 = plt.subplots(figsize=(20, 10), dpi =100)
ax1.plot(plot_data['date'], plot_data['fed_rate'], color='black', label='federal funds rate', linewidth = 2, linestyle='dashed')
ax1.plot(plot_data['date'], plot_data['shadow_rate'], color='navy', label='shadow rate', linewidth = 2, linestyle='dotted')
ax1.set_ylabel('Rate', fontsize=20) # set ylabel for different ax
ax1.set_ylim(-5, 10)
ax1.set_xlabel('Year', fontsize=20)  # need to set xlabel in ax1 or ax2
ax2 = ax1.twinx()
ax2.plot(plot_data['date'], plot_data['pred_mean'].rolling(8).mean(), color='maroon', linewidth = 2, label='level of disagreement')
ax2.set_ylabel('Level of Disagreement', fontsize=20)
ax2.set_ylim(-0.2, 0.4)
ax2.set_yticks(np.arange(0, 0.5, 0.1)) 
# get handles and labels from each subplot
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# combine labels
handles = handles1 + handles2 
labels = labels1 + labels2 
# create a single legend for the whole figure
# ncol tells python to arrange labels into num of columns, we have two labels, so if ncol=2 then we can put them horizontally
plt.legend(handles, labels, loc='upper right', ncol=3, fontsize=20)


####################################################################################
# Taylor rule regression part

result0 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + gdp_f0:pred_mean + cpi_f0:pred_mean', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result5 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + gdp_f0:pred_mean + cpi_f0:pred_mean', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result6 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result7 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result8 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + gdp_f0:pred_mean + cpi_f0:pred_mean', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result9 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result10 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result11 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + gdp_f0:pred_mean + cpi_f0:pred_mean', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')


table = summary_col([result0, result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008', 'Before2008', 'After2008', 'After2008','After2008', 'After2008_shadow', 'After2008_shadow', 'After2008_shadow'], regressor_order= [
    'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 'pred_mean', 'gdp_f0:pred_mean' , 'cpi_f0:pred_mean' 
    ], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
print(table)


#convert the table to LaTeX
latex_table = table.as_latex()
#print the LaTeX code
print(latex_table)


# result0 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + gdp_f0:pred_std + cpi_f0:pred_std', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result5 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + gdp_f0:pred_std + cpi_f0:pred_std', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result6 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result7 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result8 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + gdp_f0:pred_std + cpi_f0:pred_std', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result9 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result10 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result11 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + gdp_f0:pred_std + cpi_f0:pred_std', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')


# table = summary_col([result0, result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008', 'Before2008', 'After2008', 'After2008','After2008', 'After2008_shadow', 'After2008_shadow', 'After2008_shadow'], regressor_order= [
#     'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 'pred_std', 'gdp_f0:pred_std' , 'cpi_f0:pred_std' 
#     ], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
# print(table)

# #convert the table to LaTeX
# latex_table = table.as_latex()
# #print the LaTeX code
# print(latex_table)


# result0 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std ', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std ', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result5 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std ', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result6 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result7 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std ', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# table = summary_col([result0, result1, result2, result3, result4, result5, result6, result7], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008', 'After2008', 'After2008', 'After2008_shadow', 'After2008_shadow'], regressor_order= [
#     'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 'vote_mean', 'vote_std' 
#     ], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
# print(table)

# #convert the table to LaTeX
# latex_table = table.as_latex()
# #print the LaTeX code
# print(latex_table)

#######################################################################################
# add a dummy for 1993
# On October 5, 1993, during the FOMC conference call, Greenspan first revealed the existence of old transcripts.
# now the subsample after 2008 is useless because the 1993 dummy is always 1

result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + pred_mean:dummy_1993', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean + vote_mean:dummy_1993', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_mean + pred_mean:dummy_1993', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_mean + vote_mean:dummy_1993', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

table = summary_col([result1, result2, result3, result4], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008'], regressor_order= [
    'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 
    'pred_mean', 'pred_mean:dummy_1993', 'vote_mean', 'vote_mean:dummy_1993'], 
    info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
print(table)


#convert the table to LaTeX
latex_table = table.as_latex()
#print the LaTeX code
print(latex_table)



# result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + pred_std:dummy_1993', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std + vote_std:dummy_1993', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + pred_std + pred_std:dummy_1993', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
# result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + vote_std + vote_std:dummy_1993', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')

# table = summary_col([result1, result2, result3, result4], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008'], regressor_order= [
#     'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 
#     'pred_std',  'pred_std:dummy_1993', 'vote_std', 'vote_std:dummy_1993'], 
#     info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
# print(table)


# #convert the table to LaTeX
# latex_table = table.as_latex()
# #print the LaTeX code
# print(latex_table)

# # calculate RMSE
# from statsmodels.tools.eval_measures import rmse
# data = reg_data2
# result = result1
# y_pred = result.predict()
# rmse_value = rmse(data['fed_rate'], y_pred)
# rmse_value


########################################################################################################################################
# Part 3-B: taylor rule with speech disagreement score
########################################################################################################################################

#### voting data should be earlier than fed_rate data
voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_data.loc[voting_data['vote']==2, 'vote']=1
voting_data['date'] = pd.to_datetime(voting_data['meeting'].str[4:12])
voting_pred = voting_data[['meeting', 'pred', 'vote']].copy()
voting_pred = voting_pred.groupby(['meeting'], as_index=False).agg({'pred': ['mean','std'], 'vote': ['mean', 'std']})
voting_pred.columns = ['meeting', 'pred_mean','pred_std', 'vote_mean', 'vote_std']
voting_pred['date'] = pd.to_datetime(voting_pred['meeting'].str[4:12])
voting_pred = voting_pred[['date', 'pred_mean', 'pred_std', 'vote_mean', 'vote_std']]

# add speech data, speech data should be connected to the nearest upcoming meeting data
speech_data = pd.read_feather('parsed_speech.ftr')
speech_chunks_pred = pd.read_feather('speech_chunks_with_pred.ftr')
group_data=speech_chunks_pred.groupby(['link'], as_index=False).agg({'pred': ['mean']})
group_data.columns=['link', 'speech_pred']
speech_data = pd.merge(speech_data, group_data, on=['link'], how='left', sort=False)
speech_data = speech_data[['title', 'date', 'speaker', 'speech_pred']]
speech_data = speech_data.sort_values(by=['date']).reset_index(drop=True)
# only keep last name
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr.', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].str.replace(', Jr', '', regex=False)
speech_data['speaker'] = speech_data['speaker'].apply(lambda x: x.strip().split(' ')[-1])
speech_data['speaker'] = speech_data['speaker'].str.capitalize()  # only keep the first character capitalized
speech_data = speech_data[['date', 'speaker', 'speech_pred']].copy()

# match speech to the nearest upcoming meeting
speech_data['related_meeting'] = np.nan
voting_info = voting_data[['meeting', 'date']].copy()
voting_info = voting_info.drop_duplicates()
voting_info = voting_info.sort_values(by=['date']).reset_index(drop=True)
for i in range(1, voting_info.shape[0]):
    speech_data.loc[(speech_data['date'] >= voting_info.loc[i-1, 'date']) & (speech_data['date'] < voting_info.loc[i, 'date']), 'related_meeting'] = voting_info.loc[i, 'meeting']

speech_data = speech_data.dropna()
speech_data = speech_data[['related_meeting', 'speaker', 'speech_pred']].copy()
speech_data = speech_data.drop_duplicates()
# if a member delivered multiple speeches, we calculate the average
speech_data = speech_data.groupby(['related_meeting', 'speaker'], as_index=False).agg({'speech_pred': ['mean']})
speech_data.columns = ['meeting', 'name', 'speech_pred']

speech_pred = speech_data.groupby(['meeting'], as_index=False).agg({'speech_pred': ['mean','std']})
speech_pred.columns = ['meeting', 'speech_pred_mean','speech_pred_std']
speech_pred['date'] = pd.to_datetime(speech_pred['meeting'].str[4:12])


# combine all FOMC related data together
cb_df = [voting_pred, speech_pred, macro_data]
fomc_data = reduce(lambda left,right: pd.merge(left,right,on=['date'], how='left'), cb_df)
fomc_data.rename(columns={'date': 'meeting_date'}, inplace=True)

# now we match FOMC data timeline with fed_rate data timeline
# here we use backward merge, so fed_rate_quarter will be matched to the nearest meeting/greenbook release that was hold right before the fed_rate released.
reg_data_total = pd.merge_asof(fed_rate_quarter, fomc_data, left_on='date', right_on='meeting_date', direction='backward')
# after 2018-02-01, we do not have voting_pred data avaiable
reg_data_total.loc[reg_data_total['date'] > '2018-02-01', ['meeting_date', 'pred_std', 'pred_mean', 'vote_mean', 'vote_std', 'gdp_f0', 'gdp_f1', 'cpi_f0', 'cpi_f1', 'cpi_b1']] = np.nan
reg_data_total.drop(columns=['meeting_date'], inplace=True)


# reg_data_total.to_feather('taylor_reg_data_origin_rate_w_speech.ftr')
# reg_data_total.to_feather('taylor_reg_data_shadowrate_w_speech.ftr')

# Post-Volcker
reg_data2 = reg_data_total.loc[(reg_data_total['date'] >='1987-09-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data2.dropna(inplace=True)
# Before 2008
reg_data3 = reg_data_total.loc[(reg_data_total['date'] >='1987-09-01') & (reg_data_total['date'] <'2008-01-01'), :].copy()
reg_data3.dropna(inplace=True)
# After 2008, wo shadowrate
reg_data4 = reg_data_total.loc[(reg_data_total['date'] >='2008-01-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data4.dropna(inplace=True)
# After 2008, with shadowrate
#reg_data5 = pd.read_feather('taylor_reg_data_shadowrate.ftr')
reg_data5 = pd.read_feather('taylor_reg_data_shadowrate_w_speech.ftr')
reg_data5 = reg_data5.loc[(reg_data_total['date'] >='2008-01-01') & (reg_data_total['date'] <'2018-01-01'), :].copy()
reg_data5.dropna(inplace=True)

result0 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result1 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result2 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean + gdp_f0:speech_pred_mean + cpi_f0:speech_pred_mean', data = reg_data2, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result3 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result4 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result5 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean + gdp_f0:speech_pred_mean + cpi_f0:speech_pred_mean', data = reg_data3, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result6 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result7 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result8 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean + gdp_f0:speech_pred_mean + cpi_f0:speech_pred_mean', data = reg_data4, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result9 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result10 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')
result11 = smf.ols(formula='fed_rate ~ fed_rate_lag1 + fed_rate_lag2 + gdp_f0 + cpi_f0 + speech_pred_mean + gdp_f0:speech_pred_mean + cpi_f0:speech_pred_mean', data = reg_data5, missing = 'drop').fit().get_robustcov_results(cov_type='HC1')


table = summary_col([result0, result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11], stars=True, float_format='%.3f', model_names=['Post-Volcker', 'Post-Volcker', 'Post-Volcker', 'Before2008', 'Before2008', 'Before2008', 'After2008', 'After2008','After2008', 'After2008_shadow', 'After2008_shadow', 'After2008_shadow'], regressor_order= [
    'fed_rate', 'fed_rate_lag1', 'fed_rate_lag2', 'gdp_f0', 'cpi_f0', 'pred_mean', 'speech_pred_mean', 'gdp_f0:pred_mean' , 'gdp_f0:speech_pred_mean','cpi_f0:pred_mean', 'cpi_f0:speech_pred_mean'
    ], info_dict={"obs." : lambda x: '{0:.0f}'.format(x.nobs)})
print(table)


#convert the table to LaTeX
latex_table = table.as_latex()
#print the LaTeX code
print(latex_table)