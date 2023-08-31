# this file contructs FOMC meeting level variables

import pandas as pd
import numpy as np
from functools import reduce
from scipy.stats import entropy
# tealbook dataset: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/philadelphia-data-set
# region info data: https://www.bls.gov/cex/csxgeography.htm

# data fiels root path, make the code compatiable with mac and windows
root_path = os.path.dirname(__file__)
root_path = root_path.split('FOMC_project')[0]
file_path = root_path + 'FOMC_project/data_file/'

# (1) build dataset for each fomc member
#############################################################################
# generate region info
regions = pd.DataFrame({
    'Region': ['Northeast', 'Northeast', 'Midwest', 'Midwest', 'South', 'South', 'South', 'West', 'West'], 
    'Division': ['New England', 'Middle Atlantic', 'East North Central', 'West North Central', 'South Atlantic', 'East South Central', 'West South Central', 'Mountain', 'Pacific'],
    'States': [
        ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont'],
        ['New Jersey', 'New York', 'Pennsylvania'],
        ['Indiana', 'Illinois', 'Michigan', 'Ohio', 'Wisconsin'],
        ['Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota'],
        ['Delaware', 'D.C.', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia', 'West Virginia'],
        ['Alabama', 'Kentucky', 'Mississippi', 'Tennessee'],
        ['Arkansas', 'Louisiana', 'Oklahoma', 'Texas'],
        ['Arizona', 'Colorado', 'Idaho', 'New Mexico', 'Montana', 'Utah', 'Nevada', 'Wyoming'],
        ['Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
    ]
    })

regions = regions.explode('States')

#############################################################################
# meeting info
meeting_data = pd.read_feather('parsed_labeled_dataset.ftr')
meeting_data = meeting_data.loc[:, ['meeting', 'name', 'vote']]
meeting_member_list = pd.DataFrame({'last_name': meeting_data['name'].drop_duplicates(), 'meeting_member':1})

#############################################################################
# fomc member info
# fomc_members_full_edited.csv is manually collected
person_data = pd.read_csv(file_path+'fomc_members_full_edited.csv')
person_data['term_begin'] = person_data['period'].map(lambda x: x.split('-')[0]).astype(int)
person_data['term_end'] = person_data['period'].map(lambda x: x.split('-')[1])
person_data.loc[0:16, 'term_end'] = '2023'
person_data['term_end'] = person_data['term_end'].astype(int)
person_data['last_name'] = person_data['name'].str.replace('Jr.', '', regex = False).str.strip()
person_data['last_name'] = person_data['last_name'].apply(lambda x: x.split(' ')[-1].capitalize())
person_data['role'] = person_data['affiliation'].apply(lambda x: x.split(' ')[0])
person_data['role'] = person_data['role'].str.replace('Board', 'governor', regex=False)
person_data['role'] = person_data['role'].str.replace('Federal', 'president', regex=False)

# combine meeting info and fomc member info, only keep people who show up in both datasets (97 members)
data = pd.merge(person_data, meeting_member_list, how='inner', on='last_name', sort=False)
data = data[['last_name', 'affiliation', 'role', 'gender', 'birth', 'hometown', 'term_begin', 'term_end', 'bachelor_school', 'bachelor_subject', 'master_school', 'master_subject', 'phd_school', 'phd_subject']].copy()

data.replace('\xa0', ' ', regex = True, inplace=True)

data['hometown'] = data['hometown'].fillna('unknown')
data['hometown'] = data['hometown'].apply(lambda x: [item for item in x.split(',') if item !='' and item !=' '][-1])
data['hometown'] = data['hometown'].str.strip()
data['hometown'] = data['hometown'].str.replace('.', '', regex = False)
data['hometown'] = data['hometown'].str.replace('DC', 'D.C.', regex = False)
data['hometown'] = data['hometown'].str.replace('Minn$', 'Minnesota', regex = True)
data['hometown'] = data['hometown'].str.replace('(Brooklyn|New York City)', 'New York', regex = True)
# Melzer's hometown is unknown, some others' are outside of the US, label them as Region_Other = 1
data = pd.merge(data, regions, left_on='hometown', right_on='States', how='left', sort=False)

data = data.drop(columns=['hometown']) # delete redundant column
data=data.rename(columns={'States':'hometown_state', 'Region':'hometown_region', 'Division':'hometown_division'})
data[['hometown_region', 'hometown_division', 'hometown_state']] = data[['hometown_region', 'hometown_division', 'hometown_state']].fillna('Other') # convert those foreign places to Other

#############################################################################
# get highest school and suject names
data['highest_school'] = ''
data['highest_major'] = ''
data.replace('na', np.nan, inplace=True)
for i in range(data.shape[0]):
    if pd.isna(data.loc[i, 'phd_school']) == False:
        data.loc[i, 'highest_school'] = data.loc[i, 'phd_school']
        data.loc[i, 'highest_major'] = data.loc[i, 'phd_subject']
    elif pd.isna(data.loc[i, 'master_school']) == False:
        data.loc[i, 'highest_school'] = data.loc[i, 'master_school']
        data.loc[i, 'highest_major'] = data.loc[i, 'master_subject']
    else:
        data.loc[i, 'highest_school'] = data.loc[i, 'bachelor_school']
        data.loc[i, 'highest_major'] = data.loc[i, 'bachelor_subject']

data['highest_school'] = data['highest_school'].str.strip()
data['highest_major'] = data['highest_major'].str.strip()
# clean up the school names
data.replace('Georgetown', 'Georgetown University', regex = False, inplace=True)
data.replace('(Harvard School|Harvard University)', 'Harvard', regex = True, inplace=True)
data.replace('(Massachusetts Institute of Technology|MIT Sloan School of Management)', 'MIT', regex = True, inplace=True)
data.replace('Indiana University, Bloomington', 'Indiana University', regex = False, inplace=True)
data.replace('University of California, Berkeley', 'UC Berkeley', regex = False, inplace=True)
data.replace('(University of California at Los Angeles|University of California, Los Angeles)', 'UCLA', regex = True, inplace=True)
data.replace('Louisiana State University, Baton Rouge', 'Louisiana State University', regex = False, inplace=True)
data.replace('University of Michigan, Ann Arbor', 'University of Michigan', regex = False, inplace=True)
data.replace('University of Missouri-Kansas City.', 'University of Missouri', regex = False, inplace=True)
data.replace('(University of Wisconsin–Madison|University of Wisconsin, Madison)', 'University of Wisconsin', regex = True, inplace=True)
data.replace('Wharton School of the University of Pennsylvania', 'University of Pennsylvania', regex = False, inplace=True)
data.replace('(yale|Yale University)', 'Yale', regex = True, inplace=True)
data['highest_school'] = data['highest_school'].str.replace('.', '', regex =False)
# simplify major names
sub = {
    'Economics':'economics',
    'finance':'economics',
    'agriculture and economics':'economics',
    'economics.':'economics',
    'economics and public administration':'economics',
    'monetary economics':'economics',
    'political economy':'economics',
    'finance and business economics':'economics',
    'finance and economics':'economics',
    'business and applied economics':'economics',
    'international economics and East Asian studies':'economics',
    'international economics and U.S. foreign policy':'economics',
    'banking':'economics',
    'Law':'law',
    'business administration':'management',
    'business administration in finance':'management',
    'mba':'management',
    'business adminstration':'management',
    'Business Administration':'management',
    'MBA':'management',
    'political science':'other',
    'civil engineering':'other',
    'industrial management':'other'
}

data['highest_major'] = data['highest_major'].apply(lambda x: sub.get(x, x))
data['phd_school'] = data['phd_school'].fillna('other')
data['phd_subject'] = data['phd_subject'].fillna('other')
data['phd_subject'] = data['phd_subject'].apply(lambda x: sub.get(x, x))
data=data.rename(columns={'phd_subject':'phd_major'})

data = data.drop(columns=['bachelor_school', 'bachelor_subject', 'master_school', 'master_subject']) # delete redundant column

#############################################################################
# add univ. categories

univ_info = pd.read_csv(file_path+'univ_cat.csv')
univ_info = univ_info[['Name', 'Location', 'End_per_Stu']].copy()
univ_info.rename(columns={'Name': 'school_name', 'Location':'school_location'}, inplace=True)

# build region info for highest school
data = pd.merge(data, univ_info, left_on='highest_school', right_on='school_name', how='left', sort=False)
data.rename(columns={'school_location': 'highest_school_location', 'End_per_Stu': 'highest_school_end_per_stu'}, inplace=True)
data.drop(columns='school_name', inplace=True)
data = pd.merge(data, regions, left_on='highest_school_location', right_on='States', how='left', sort=False)
data.rename(columns={'Region': 'highest_school_region', 'Division':'highest_school_division'}, inplace=True)
data.drop(columns='States', inplace=True)

# build region info for phd school
data = pd.merge(data, univ_info, left_on='phd_school', right_on='school_name', how='left', sort=False)
data.rename(columns={'school_location': 'phd_school_location', 'End_per_Stu': 'phd_school_end_per_stu'}, inplace=True)
data.drop(columns='school_name', inplace=True)
data = pd.merge(data, regions, left_on='phd_school_location', right_on='States', how='left', sort=False)
data.rename(columns={'Region': 'phd_school_region', 'Division':'phd_school_division'}, inplace=True)
data.drop(columns='States', inplace=True)
data[['phd_school_location', 'phd_school_region', 'phd_school_division']] = data[['phd_school_location', 'phd_school_region', 'phd_school_division']].fillna('other')
data['phd_school_end_per_stu'] = data['phd_school_end_per_stu'].fillna(0)

#############################################################################
# add event info
# WWI and the following inflation: 1914-1918
# Great Depression: 1929-1939
# WWII: 1939-1945
# Great Inflation: 1965-1982

# we want to see if the person experience these events before they turn age_threshold = 21
age_threshold = 21
data['wwi'] = 1
data['wwii'] = 1
data['great_depression'] = 1
data['great_inflation'] = 1
for i in range(data.shape[0]):
    if (data.loc[i, 'birth'] + age_threshold <1914) or (data.loc[i, 'birth'] > 1918) :
        data.loc[i, 'wwi'] = 0
    if (data.loc[i, 'birth'] + age_threshold <1929) or (data.loc[i, 'birth'] > 1939) :
        data.loc[i, 'great_depression'] = 0
    if (data.loc[i, 'birth'] + age_threshold <1939) or (data.loc[i, 'birth'] > 1945) :
        data.loc[i, 'wwii'] = 0
    if (data.loc[i, 'birth'] + age_threshold <1965) or (data.loc[i, 'birth'] > 1982) :
        data.loc[i, 'great_inflation'] = 0


#############################################################################
# generate potus info
# we want to see the potus party affilication who appoint them 

# for presidents who are not appointed by presidents directly, we use their districts presidential election results instead of president to label the party
# the presidential election results by states is downloaded from MIT Election Data and Science Lab: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/42MVDX
# MIT Election Data and Science Lab https://electionlab.mit.edu/data


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


# state level presidential election results
# potus_party here means the president candidate who won the most vote in this state, not the real potus
bank_locations = pd.DataFrame(
    {'bank': ['Federal Reserve Bank of Boston', 'Federal Reserve Bank of New York', 'Federal Reserve Bank of Philadelphia', 'Federal Reserve Bank of Cleveland', 'Federal Reserve Bank of Richmond', 'Federal Reserve Bank of Atlanta', 'Federal Reserve Bank of Chicago', 'Federal Reserve Bank of St. Louis', 'Federal Reserve Bank of Minneapolis', 'Federal Reserve Bank of Kansas City', 'Federal Reserve Bank of Dallas', 'Federal Reserve Bank of San Francisco'],
    'state': ['MA', 'NY', 'PA', 'OH', 'VA', 'GA', 'IL', 'MO', 'MN', 'MO', 'TX', 'CA']}
    )

state_vote = pd.read_csv(file_path+'1976-2020-presidential_election.csv')
# pick out the winning party in each state
state_vote = state_vote.loc[state_vote.groupby(['year', 'state'])['candidatevotes'].idxmax()]
state_vote = state_vote.loc[state_vote['state_po'].isin(bank_locations['state'])] # only keep certain states

state_vote = state_vote[['year', 'state_po', 'party_simplified']]
state_vote.columns = ['year', 'state', 'potus_party']

# add early election results
# note: in 1968, GA voted for independent
early_sate_vote = pd.DataFrame(
    {'year': [1968]*11 + [1972]*11, 
    'state': ['CA', 'GA', 'IL', 'MA', 'MN', 'MO', 'NY', 'OH', 'PA', 'TX', 'VA']*2,
    'potus_party': ['REPUBLICAN', 'INDEPENDENT', 'REPUBLICAN', 'DEMOCRAT', 'DEMOCRAT', 'REPUBLICAN', 'DEMOCRAT', 'REPUBLICAN', 'DEMOCRAT', 'DEMOCRAT', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'DEMOCRAT', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN', 'REPUBLICAN']
    }
)

state_vote = pd.concat([early_sate_vote, state_vote], axis=0, ignore_index=True)
state_vote['potus_party'] = state_vote['potus_party'].str.capitalize()

year = []
state = []
party = []
for i in range(state_vote.shape[0]):
    elect_year = state_vote.loc[i, 'year']
    year = year + [elect_year-1, elect_year, elect_year+1, elect_year+2]
    state = state + [state_vote.loc[i, 'state']] * 4
    party = party + [state_vote.loc[i, 'potus_party']] * 4
state_vote_list = pd.DataFrame({'year': year, 'state': state, 'potus_party': party})

state_vote_list = pd.merge(state_vote_list, bank_locations, on='state', how='left', sort=False)
state_vote_list = state_vote_list[['year', 'bank', 'potus_party']]
potus_list['bank'] = 'Board of Governors'
potus_list = potus_list[['year', 'bank', 'potus_party']]
potus_list = pd.concat([potus_list, state_vote_list], axis=0, ignore_index=True)

data = pd.merge(data, potus_list, left_on=['term_begin', 'affiliation'], right_on=['year', 'bank'], how='left', sort=False)
data.drop(columns=['year', 'bank'], inplace=True)

#########################################################################################################################
#########################################################################################################################
# (2) build the dataset for meetings
meeting_data['date'] = pd.to_datetime(meeting_data['meeting'].str[4:12])
meeting_data['year'] = meeting_data['date'].dt.year
unique_meetings = meeting_data['meeting'].unique()

# from 2004–2010, Janet L. Yellen was President, Federal Reserve Bank of San Francisco, other times she is a governor
# from 1975-1979, Paul A. Volcker was President, Federal Reserve Bank of New York, other times he was a governor
experience = []            # experience std in that meeting
age = []                   # age std in that meeting 
gender = []                # the percentage of female
hometown_region = []       # hometown region diversity
hometwon_entropy = []      # hometown region entropy
highest_school_region = [] # highest school region diversity 
highest_school_entropy = [] # highest school region entropy
highest_school_wealth = [] # highest school endowment per student std
highest_major = []         # highest major is econ or not
wwi = []                   # percentage of member experienced wwi
wwii = []                  # percentage of member experienced wwii
depression_dis = []        # percentage of member experienced great depression
inflation_dis = []         # percentage of member experienced great inflation
app_potus_party = []       # the percentage of democrat
app_potus_entropy = []     # the binary entropy of the appointing potus's party distribution, [0.5, 0.5] has the highest entropy (1), [0,1] or [1,0] has the lowest (0)

for meet in unique_meetings:
    temp = meeting_data.loc[meeting_data['meeting']==meet].copy()
    temp = pd.merge(temp[['meeting','name', 'year']], data, left_on='name', right_on='last_name', how='left', sort=False)
    temp = temp.dropna()

    # experience distribtution
    temp['experience'] = temp['year'] - temp['term_begin']
    experience.append(np.std(temp['experience']))

    # age distribution
    temp['age'] =  temp['year'] - temp['birth']
    age.append(np.std(temp['age']))

    # gender ratio
    gender_temp = temp['gender'].value_counts()
    if len(gender_temp) == 1:
        gender.append(0)
    else:
        gender.append(gender_temp['female']/temp.shape[0])

    # hometown region distritbution
    ht_temp = temp['hometown_region'].value_counts()
    # hometown region entropy
    ht_entropy_temp = [x/temp.shape[0] for x in ht_temp]
    ht_entropy_temp = entropy(ht_entropy_temp, base=5)
    hometwon_entropy.append(round(ht_entropy_temp,6))

    # highest school region distritbution
    school_temp = temp['highest_school_region'].value_counts()
    # highest school region entropy
    school_entroy_temp = [x/temp.shape[0] for x in school_temp]
    school_entroy_temp = entropy(school_entroy_temp, base=4)
    highest_school_entropy.append(round(school_entroy_temp,6))

    # highest school wealth distribution
    highest_school_wealth.append(np.std(temp['highest_school_end_per_stu']))

    # highest major, the percentage of econ major
    highest_major.append(temp.loc[temp['highest_major']=='economics'].shape[0]/temp.shape[0])

    # experience wwi, wwii, great depression, great inflation ratio
    wwi.append(temp['wwi'].sum()/temp.shape[0])
    wwii.append(temp['wwii'].sum()/temp.shape[0])
    depression_dis.append(temp['great_depression'].sum()/temp.shape[0])
    inflation_dis.append(temp['great_inflation'].sum()/temp.shape[0])

    # when they first appointed, the sitting potus party distritbution
    try:
        app_potus_party.append(temp['potus_party'].value_counts()['Democrat']/temp.shape[0])
    except:
        app_potus_party.append(0)

    # calculate the appointing potus party distributuion entropy (binary entropy)
    try:
        entropy_value = entropy([temp['potus_party'].value_counts()['Democrat']/temp.shape[0], temp['potus_party'].value_counts()['Republican']/temp.shape[0]], base=2)
        app_potus_entropy.append(entropy_value)
    except:
        app_potus_entropy.append(0)


reg_data = pd.DataFrame({'meeting': unique_meetings,'experience': experience, 'age': age, 'gender': gender, 'hometown_region': hometown_region, 'hometown_entropy': hometwon_entropy, 'highest_school_region': highest_school_region, 'highest_school_entropy': highest_school_entropy,  'highest_school_wealth': highest_school_wealth, 'highest_major': highest_major, 'wwi': wwi, 'wwii': wwii, 'depression_dis': depression_dis, 'inflation_dis': inflation_dis, 'app_potus_party': app_potus_party, 'app_potus_entropy': app_potus_entropy})

reg_data['date'] = pd.to_datetime(reg_data['meeting'].str[4:12])
reg_data = reg_data.sort_values(by='date', ascending=True).reset_index(drop=True)


# add gdp, inflation and unemployment info from the teal book
def extract_trend(y):
    try:
        y = y.dropna()
        x = np.array(range(len(y)))
        trend, intercept = np.polyfit(x=x, y=y, deg=1)
        return trend
    except:
        return np.nan


unemploy =  pd.read_csv(file_path+'tealbook_dataset/row_format/unemploy.csv')
unemploy = unemploy.loc[unemploy['meeting_date'].isna() == False, :]
unemploy = unemploy[['meeting_date', 'UNEMPB2', 'UNEMPB1', 'UNEMPF0', 'UNEMPF1', 'UNEMPF2']].copy()
unemploy['unemploy_trend'] = unemploy.apply(lambda x: extract_trend(x[1:].dropna().astype(float)), axis=1)
unemploy['unemploy_std'] = unemploy.apply(lambda x: np.nanstd(x[1:].astype(float)), axis=1)
unemploy.drop(columns=['UNEMPB2', 'UNEMPB1', 'UNEMPF1', 'UNEMPF2'], inplace=True)
unemploy.rename(columns={'UNEMPF0': 'unemploy', 'meeting_date': 'date'}, inplace=True)
unemploy['date'] = pd.to_datetime(unemploy['date'])


core_cpi = pd.read_csv(file_path+'tealbook_dataset/row_format/core_cpi.csv')
core_cpi = core_cpi.loc[core_cpi['meeting_date'].isna() == False, :]
core_cpi = core_cpi[['meeting_date', 'gPCPIXB2', 'gPCPIXB1', 'gPCPIXF0', 'gPCPIXF1', 'gPCPIXF2']].copy()
core_cpi['core_cpi_trend'] = core_cpi.apply(lambda x: extract_trend(x[1:].dropna().astype(float)), axis=1)
core_cpi['core_cpi_std'] = core_cpi.apply(lambda x: np.nanstd(x[1:].astype(float)), axis=1) # std omitting nan
core_cpi.drop(columns=['gPCPIXB2', 'gPCPIXB1', 'gPCPIXF1', 'gPCPIXF2'], inplace=True)
core_cpi.rename(columns={'gPCPIXF0': 'core_cpi', 'meeting_date': 'date'}, inplace=True)
core_cpi['date'] = pd.to_datetime(core_cpi['date'])

# combine data
cb_df = [reg_data, gdp, unemploy, headline_cpi, core_cpi]
reg_data = reduce(lambda left,right: pd.merge(left,right,on=['date'], how='left'), cb_df)
# new tealbook is not provided in conference calls, so in each confrence call we can use the same macro data in the nearest previous meeting.
reg_data[['unemploy', 'unemploy_trend', 'unemploy_std', 'core_cpi', 'core_cpi_trend', 'core_cpi_std']] = reg_data[['unemploy', 'unemploy_trend', 'unemploy_std', 'core_cpi', 'core_cpi_trend', 'core_cpi_std']].fillna(method='ffill', axis=0)

# add the sitting president's party info
reg_data['year'] = reg_data['date'].dt.year
reg_data = pd.merge(reg_data, potus_list.loc[potus_list['bank']=='Board of Governors', ['year', 'potus_party']], on=['year'], how='left')
reg_data['current_potus_party'] = 0
reg_data.loc[reg_data['potus_party']=='Democrat', 'current_potus_party']=1
reg_data.drop(columns=['potus_party', 'year'], inplace=True)

reg_data.to_feather('fomc_info_reg_data.ftr')


