import pandas as pd
import numpy as np
from pypdf import PdfReader
import re
import os

# transcripts are downloaded from https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm

# extract all the file names in the folder
root_path = os.path.dirname(__file__)
root_path = re.split('Dropbox', root_path)[0]
file_path = root_path + 'Dropbox/transcript_original_files/'
files = os.listdir(file_path)
files.remove('.DS_Store')  # remove the hidden file


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
            
            text_content=pdfreader.pages[page].extract_text()
            if 'Transcript' in text_content:
                start_page_index = page
                break


        # extract data 
        names = []
        data = []
        for page in range(start_page_index, page_num):
            
            # extract the raw data
            text_content=pdfreader.pages[page].extract_text()

            # parse the raw data
            # clean raw data
            text_content = text_content.strip() 
            text_content = re.sub('\\n\d+', ' ', text_content) 
            # remove page header info
            text_content = re.sub('\s{0,3}[A-Z]\w+\s+\d+.?\d*,\s*\d+\s+of\s+\d+', '',text_content).strip() 
            text_content = re.sub('\s{0,3}(\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2}\s?-\s?\d{1,2}|\d{1,2}/\d{1,2}\s?-\s?\d{1,2}/\d{1,2})/\d{2,4}', ' ', text_content).strip()  
            text_content = re.sub('\s{0,3}\d+\\n', ' ', text_content).strip() 
            text_content = re.sub('-\s?\d{1,3}\s?-', ' ', text_content).strip()
            # remove all '\n'
            text_content = re.sub('(\n|\xa0|\xad)', ' ', text_content).strip()
            # remove [Applause], [Laughter] and [Unintelligible], [Statement--see Appendix.]
            text_content = re.sub('\[(A|a)pplause\.?\]', '', text_content).strip() 
            text_content = re.sub('\[(L|l)aughter\.?\]', '', text_content).strip()
            text_content = re.sub('\[(U|u)nintelligible\.?\]', '...', text_content).strip()
            text_content = re.sub('\[(S|s)tatement\s?--\s?see\s{0,2}(A|a)ppendix\.?\]', '...', text_content).strip()
            # remove []
            text_content = re.sub('\[', '', text_content).strip()
            text_content = re.sub('\]', '', text_content).strip()
            # remove session info
            text_content = re.sub('[A-Z]\w+\s+\d+.+Session', '', text_content).strip() 
            
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
            
            # correct spelling mistakes for GREENSPAN, MADONOUGH, BERNANKE, YELLEN, DUDLEY, GEITHNER, VOLCKER, MILLER
            text_content = re.sub('G\s{0,2}R\s{0,2}E\s{0,2}E\s{0,2}N\s{0,2}S\s{0,2}P\s{0,2}A\s{0,2}N', 'GREENSPAN', text_content).strip()
            text_content = re.sub('M\s{0,2}C\s{0,2}D\s{0,2}O\s{0,2}N\s{0,2}O\s{0,2}U\s{0,2}G\s{0,2}H', 'MCDONOUGH', text_content).strip()
            text_content = re.sub('B\s{0,2}E\s{0,2}R\s{0,2}N\s{0,2}A\s{0,2}N\s{0,2}K\s{0,2}E', 'BERNANKE', text_content).strip()
            text_content = re.sub('Y\s{0,2}E\s{0,2}L\s{0,2}L\s{0,2}E\s{0,2}N', 'YELLEN', text_content).strip()
            text_content = re.sub('D\s{0,2}U\s{0,2}D\s{0,2}L\s{0,2}E\s{0,2}Y', 'DUDLEY', text_content).strip()
            text_content = re.sub('(GREESPAN|GREENPAN|GREENPSAN|GRENSPAN)', 'GREENSPAN', text_content).strip()
            text_content = re.sub('GEITHER', 'GEITHNER', text_content).strip()
            text_content = re.sub('(VOLKER|VOLCRER)', 'VOLCKER', text_content).strip() 
            text_content = re.sub('(MILER|MJLLER|MILIER)', 'MILLER', text_content).strip() 
            text_content = re.sub('O\s{0,2}’\s{0,2}CONNELL', 'OCONNELL', text_content).strip()
            # there are MR. LINDSEY and MR. D.LINDSEY, use DLINDSEY to represent D. LINDSEY
            text_content = re.sub('D\s{0,2}\.\s{0,2}LINDSEY', 'DLINDSEY', text_content).strip()

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
        # remove the '.' at the end of the string
        final_data['name']= final_data['name'].replace(regex=r'\.$', value ='')
        # remove footnote number
        final_data['words']= final_data['words'].replace(regex=r'^\d{1,2}', value = '')
        # combine the words from the same person together
        final_data = final_data.groupby(['name'])['words'].apply(' '.join).reset_index()
        # remove the '.' at the beginning of the string
        final_data['words']= final_data['words'].replace(regex=r'^\.', value = '')

        
        feather_path = 'parsed_raw_data/'+file_name+'.ftr'
        final_data.to_feather(feather_path)

      
    
    except:
        print('ERROR:', file_name)


