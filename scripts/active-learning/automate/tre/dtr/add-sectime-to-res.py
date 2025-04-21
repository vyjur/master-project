import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime
import pypdf
import pikepdf
from textmining.tee.setup import TEExtract

journal_path = "./data/helsearkiv/journal/"

in_path = "./data/helsearkiv/batch/dtr/"

out_path = "./data/helsearkiv/batch/dtr-w-dct/"

batch_files = [
    f
    for f in os.listdir(in_path)
    if os.path.isfile(os.path.join(in_path, f)) and "final" in f 
]

manager = None

tee = TEExtract("./src/textmining/tee/config.ini", manager)

for file in batch_files:
    df = pd.read_csv(in_path + file)
    
    df['SECTIME'] = np.nan
    df['SECTIME_context'] = np.nan
    
    group_df = df.groupby(['file', 'page'])
    
    for (file_val, page_val), group in group_df:
        
        reader = pypdf.PdfReader(journal_path + file_val)
        doc = reader.pages[int(page_val)].extract_text()
        dcts, sectimes = tee.extract_sectime(doc)

        for i, dct in enumerate(sectimes):
            entities = []
            start = sectimes[i]["index"]
            if i + 1 >= len(sectimes):
                stop = len(doc)
            else:
                stop = sectimes[i+1]["index"]
            sec_text = doc[start:stop]
            sectimes[i]['text'] = sec_text
            sectimes[i]['context'] = doc[max(0, start - 50) : min(len(doc), start+50)]
            
        for index, row in group.iterrows():
            row_text = row['Text']  # Assuming 'Text' column has the text you want to compare

            # Loop through sectimes and check if row['Text'] is within the 'text' from sectimes
            for sectime in sectimes:
                if row_text in sectime['text']:  # If the row's text is found in the section's text
                    print(f"Row {index} matches sectime: {sectime}")
                    
                    # Save the sectime['text'] to the 'SECTIME' column of the DataFrame row
                    df.at[index, 'SECTIME'] = sectime['text']
                    df.at[index, 'SECTIME_context'] = sectime['context']
                    break  # Stop once a match is found, remove if you want to find all matches
                
    df.to_csv(out_path + file)

        