# Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings - Code and Data
This folder contains all the code and necessary data sets that can replicate the paper *Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings* by Kwok Ping Tsang and Zichao Yang.

###  Data

All FOMC transcripts data can be downloaded from https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

FOMC member's speech data can be downloaded from the Federal Reserve Board (https://www.federalreserve.gov) and local Federal Reserve Banks. 

Due to the size of transcripts and speeches, they are not included in this repository. Here we only provide the processed data from the deep learning model; the data should be sufficient to replicate all the results in this paper. 

Meanwhile, python files 1-8 provide detailed instructions on how to train the deep learning model. Readers who are interested in the deep learning model can download the transcripts and speeches and train the model by themselves.

### Code

python file 1-4 : process transcript data

python file 5: train the deep learning model based on transcript data

python file 7: process speech data

python file 6, 8: evaluate transcript data and speech data based on fine-tuned deep learning model

python file 9, 10: compile FOMC-related data, file 10 only contains FOMC members who gave a speech before attending the FOMC meeting.

python file 11: replicate all figures in paper

python file 12: replicate all meeting-level regressions in paper

python file 13_a: generate data for individual level regressions

R file 13_b: replicate individual level regressions in paper

