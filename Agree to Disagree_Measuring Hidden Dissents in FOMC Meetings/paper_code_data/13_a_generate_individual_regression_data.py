import pandas as pd
import numpy as np
#import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import PooledOLS
from linearmodels.panel import compare

##########################
# voting data
# title: 1: CHAIR 2: VICE CHAIR
voting_data = pd.read_feather('org_data_w_pred_w_chair_20230407.ftr')
voting_data = voting_data[['meeting', 'name', 'title', 'pred', 'vote']].copy()
voting_data.columns = ['meeting', 'name', 'title', 'voting_pred', 'vote']

voting_data.loc[voting_data['vote']==2, 'vote'] = 1

##########################
# personal info
personal_info = pd.read_feather('fomc_member_personal_info.ftr')
personal_info = personal_info[['last_name', 'gender', 'birth', 'term_begin', 'hometown_region', 'highest_school_region', 'highest_school_end_per_stu', 'highest_major', 'wwi', 'wwii', 'great_depression', 'great_inflation', 'potus_party']].copy()
personal_info.rename(columns={'last_name': 'name', 'highest_school_end_per_stu': 'highest_school_wealth'}, inplace=True)
# personal_info['hometown_region'].value_counts()
# convert to dummy
gender_dummy = pd.get_dummies(personal_info['gender'], prefix='gender')
personal_info = pd.concat([personal_info, gender_dummy['gender_female']], axis=1)

hometwon_dummy = pd.get_dummies(personal_info['hometown_region'], prefix='hometown')
personal_info = pd.concat([personal_info, hometwon_dummy[['hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West']]], axis=1)

highest_school_dummy = pd.get_dummies(personal_info['highest_school_region'], prefix='highest_school')
personal_info = pd.concat([personal_info, highest_school_dummy[['highest_school_Northeast', 'highest_school_South', 'highest_school_West']]], axis=1)

highest_major_dummy = pd.get_dummies(personal_info['highest_major'], prefix='highest_major')
personal_info = pd.concat([personal_info, highest_major_dummy['highest_major_economics']], axis=1)

potus_dummy = pd.get_dummies(personal_info['potus_party'], prefix='potus')
personal_info = pd.concat([personal_info, potus_dummy['potus_Democrat']], axis=1)

personal_info.drop(columns=['gender', 'hometown_region', 'highest_school_region', 'highest_major', 'potus_party'], inplace=True)

##########################
# economy info

macro_info = pd.read_feather('fomc_info_reg_data.ftr')
macro_info = macro_info[['meeting', 'gdp', 'gdp_trend', 'gdp_std', 'unemploy', 'unemploy_trend', 'unemploy_std', 'headline_cpi', 'headline_cpi_trend','headline_cpi_std', 'core_cpi', 'core_cpi_trend', 'core_cpi_std']].copy()

#########################
# regression data

reg_data = pd.merge(voting_data, macro_info, on='meeting', how='left')
reg_data = pd.merge(reg_data, personal_info, on='name', how='left')
reg_data['date'] = pd.to_datetime(reg_data['meeting'].str[4:12])
reg_data['year'] = reg_data['date'].dt.year

reg_data['age'] = reg_data['year'] - reg_data['birth']
reg_data['experience'] = reg_data['year'] - reg_data['term_begin']

#########################
# the sitting potus

presidents = {
'Harry S. Truman': ['Democrat', '1945-1953'],
'Dwight D. Eisenhower': ['Republican', '1953-1961'],
'John F. Kennedy': ['Democrat', '1961-1963'],
'Lyndon B. Johnson': ['Democrat', '1963-1969'],
'Richard Nixon': ['Republican', '1969-1974'],
'Gerald Ford': ['Republican', '1974-1977'],
'Jimmy Carter': ['Democrat', '1977-1981'],
'Ronald Reagan': ['Republican', '1981-1989'],
'George H.W. Bush': ['Republican', '1989-1993'],
'Bill Clinton': ['Democrat', '1993-2001'],
'George W. Bush': ['Republican', '2001-2009'],
'Barack Obama': ['Democrat', '2009-2017'],
'Donald Trump': ['Republican', '2017-2021'],
'Joe Biden': ['Democrat', '2021-2023']
}
potus = pd.DataFrame(presidents).T
potus = potus.reset_index()
potus.columns = ['name', 'party', 'term']
potus['term_begin'] = potus['term'].map(lambda x: x.split('-')[0]).astype(int)
potus['term_end'] = potus['term'].map(lambda x: x.split('-')[1]).astype(int)

year = []
name = []
party = []
for i in range(potus.shape[0]):
    year = year + list(range(potus.loc[i, 'term_begin'], potus.loc[i, 'term_end']))
    name = name + [potus.loc[i, 'name']] * (potus.loc[i, 'term_end'] - potus.loc[i, 'term_begin'])
    party = party + [potus.loc[i, 'party']] * (potus.loc[i, 'term_end'] - potus.loc[i, 'term_begin'])
potus_list = pd.DataFrame({'potus_name': name, 'potus_party': party, 'year': year})

reg_data = pd.merge(reg_data, potus_list[['year', 'potus_party']], on=['year'], how='left')
reg_data['current_potus_party'] = 0
reg_data.loc[reg_data['potus_party']=='Democrat', 'current_potus_party']=1
reg_data.drop(columns=['potus_party'], inplace=True)

reg_data.to_feather('fomc_individual_reg_data.ftr')

# remove chair data points
# reg_data = reg_data[reg_data['title'] != 1]

