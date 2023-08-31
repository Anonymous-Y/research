library(arrow)
library(dplyr)
library(lme4)
#library(glmmTMB)
library(plm)
#library(pglm)
library(stargazer)
library(margins)

# set wd
setwd("~/Library/CloudStorage/Dropbox/A_Research_Projects/FOMC_project/Code")
setwd("D:/Dropbox/A_Research_Projects/FOMC_project/Code")

# the data is generated in 20_panel_regression_fomc_individual.py
reg_data = read_feather('fomc_individual_reg_data.ftr')

# remove chairs' data
# reg_data = reg_data[reg_data$title !=1, ]

############################################################################################
############################################################################################
# Part 1, explain disagreement and dissents
############################################################################################
############################################################################################

##################################################################################
# Linear Probability Model
##################################################################################

# Pooled OLS (aka OLS)
voting_pool = lm(voting_pred ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data)

vote_pool = lm(vote ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data)


# mixed effect OLS
voting_mixed = lmer(voting_pred ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data)

vote_mixed = lmer(vote ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data)


se1 = sqrt(diag(vcovHC(voting_pool, type = "HC1")))
se2 = sqrt(diag(vcovHC(vote_pool, type = "HC1")))
se3 = coef(summary(voting_mixed))[,'Std. Error']
se4 = coef(summary(vote_mixed))[,'Std. Error']

stargazer(voting_pool, vote_pool, voting_mixed, vote_mixed,
          align = TRUE,
          se = list(se1, se2, se3, se4),
          type = "text", star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'vote',  'disagreement', 'vote', 'disagreement', 'vote'),
          #omit = 'Bayesian Inf. Crit.',
          order = c('unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)

# latex version
stargazer(voting_pool, vote_pool, voting_mixed, vote_mixed,
          align = TRUE,
          se = list(se1, se2, se3, se4),
          star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'vote',  'disagreement', 'vote', 'disagreement', 'vote'),
          #omit = 'Bayesian Inf. Crit.',
          order = c('unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)


############################################################################################
# Logit Model
#############################################################################################


# logit 
voting_logit = glm(voting_pred ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data, family = quasibinomial(link = "logit"))
# summary(voting_logit)


vote_logit = glm(vote ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data, family = binomial(link = "logit"))
# summary(vote_logit)


# mixed effect logit

voting_mixed_logit = glmer(voting_pred ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data, family = binomial(link = "logit"))
# summary(voting_mixed)


vote_mixed_logit = glmer(vote ~ unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data, family = binomial(link = "logit"))
# summary(vote_mixed)

stargazer(voting_logit, vote_logit, voting_mixed_logit, vote_mixed_logit,
          align = TRUE,
          type = "text", star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'vote',  'disagreement', 'vote', 'disagreement', 'vote'),
          omit = 'Bayesian Inf. Crit.',
          order = c('unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)


# latex version
stargazer(voting_logit, vote_logit, voting_mixed_logit, vote_mixed_logit,
          align = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'vote',  'disagreement', 'vote', 'disagreement', 'vote'),
          omit = 'Bayesian Inf. Crit.',
          order = c('unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)

############################################################################################
############################################################################################
# Part 2: examine the relationship between speech and transcript on individual level
############################################################################################
############################################################################################

# the data is generated in 20_panel_regression_fomc_individual.py
reg_data = read_feather('fomc_individual_speech_reg_data.ftr')
reg_data = na.omit(reg_data)

# remove chairs' data
reg_data = reg_data[reg_data$title !=1, ]


# # Pooled OLS (aka OLS)
# voting_pool = lm(voting_pred ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data)
# 
# vote_pool = lm(vote ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party, data = reg_data)


# mixed effect OLS
voting_mixed_1 = lmer(voting_pred ~ speech_pred + (1 | name), data = reg_data)

# voting_mixed_2 = lmer(voting_pred ~ speech_pred + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data)
# 
# voting_mixed_3 = lmer(voting_pred ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + (1 | name), data = reg_data)

voting_mixed_4 = lmer(voting_pred ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data)

vote_mixed_1 = lmer(vote ~ speech_pred + (1 | name), data = reg_data)

vote_mixed_2 = lmer(vote ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data)



stargazer(voting_mixed_1, voting_mixed_4, vote_mixed_1, vote_mixed_2,
          align = TRUE,
          #se = list(se1, se2, se3, se4),
          type = "text", star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'disagreement', 'disagreement', 'disagreement', 'vote', 'vote'),
          #omit = 'Bayesian Inf. Crit.',
          order = c('speech_pred', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)

# latex

stargazer(voting_mixed_1, voting_mixed_4, vote_mixed_1, vote_mixed_2,
          align = TRUE,
          #se = list(se1, se2, se3, se4),
          star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'disagreement', 'vote', 'vote'),
          #omit = 'Bayesian Inf. Crit.',
          order = c('speech_pred', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)



# mixed effect logit

voting_mixed_logit_1 = glmer(voting_pred ~ speech_pred + (1 | name), data = reg_data, family = binomial(link = "logit"))

# voting_mixed_logit_2 = glmer(voting_pred ~ speech_pred + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data, family = binomial(link = "logit"))
# 
# voting_mixed_logit_3 = glmer(voting_pred ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + (1 | name), data = reg_data, family = binomial(link = "logit"))

voting_mixed_logit_4 = glmer(voting_pred ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data, family = binomial(link = "logit"))

vote_logit_1 = glmer(vote ~ speech_pred + (1 | name), data = reg_data, family = binomial(link = "logit"))

vote_logit_2 = glmer(vote ~ speech_pred + unemploy_trend + unemploy_std + core_cpi_trend + core_cpi_std + age + experience + highest_school_wealth + gender_female + hometown_Northeast + hometown_Other + hometown_South + hometown_West + highest_school_Northeast + highest_school_South + highest_school_West + highest_major_economics + wwii + great_depression + great_inflation + potus_Democrat + current_potus_party + (1 | name), data = reg_data, family = binomial(link = "logit"))


stargazer(voting_mixed_logit_1, voting_mixed_logit_4, vote_logit_1, vote_logit_2,
          align = TRUE,
          type = "text", star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'disagreement', 'disagreement', 'disagreement', 'vote', 'vote'),
          omit = 'Bayesian Inf. Crit.',
          order = c('speech_pred', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)


# latex
stargazer(voting_mixed_logit_1, voting_mixed_logit_4, vote_logit_1, vote_logit_2,
          align = TRUE,
          star.cutoffs = c(0.1, 0.05, 0.01), digits = 3, omit.stat = c("f", "ser"), 
          column.labels = c('disagreement', 'disagreement', 'disagreement', 'disagreement', 'vote', 'vote'),
          omit = 'Bayesian Inf. Crit.',
          order = c('speech_pred', 'unemploy_trend', 'unemploy_std', 'core_cpi_trend', 'core_cpi_std', 'age', 'experience', 'highest_school_wealth', 'gender_female', 'hometown_Northeast', 'hometown_Other', 'hometown_South', 'hometown_West', 'highest_school_Northeast', 'highest_school_South', 'highest_school_West', 'highest_major_economics', 'wwii', 'great_depression', 'great_inflation', 'potus_Democrat', 'current_potus_party')
)

# margins(voting_mixed_logit_4) # calculate the marginal effect
