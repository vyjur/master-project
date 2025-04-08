import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime
import pypdf
import pikepdf


file = "./data/helsearkiv/batch/dtr/2-final.csv"   
df = pd.read_csv(file)

STOP = len(df)
df = df.sort_values(by="prob", ascending=True)  # Sort in ascending order

if "TIMEX_y" in df.columns:
    df = df.drop(["TIMEX_y"], axis=1)
df = df.rename(columns={"Context_x": "Context", "TIMEX_x": "TIMEX"})

# Filter rows where TIMEX is not NaN
timex_df = df[df['TIMEX'].isna()]
print(timex_df)

print("LEN:", len(timex_df))
#writer = pypdf.PdfWriter()
#count = 0

#for i, (index, row) in enumerate(timex_df.iterrows()):
    #if i % 50 == 0:
        #for page in writer.pages:
            #page.compress_content_streams(level=9)
        #with open (f"output/main-{count}.pdf", "wb") as output_file:
            #writer.write(output_file)
        #with pikepdf.open(f'output/main-{count}.pdf', allow_overwriting_input=True) as pdf:
            #pdf.save(f'output/main-{count}.pdf', compress_streams=True)
        #writer = pypdf.PdfWriter()
        #count += 1
        
    #if i >= 4000:
        #break
    #if pd.isna(row['file']):
        #continue
    #reader = pypdf.PdfReader('./data/helsearkiv/journal/' + str(row['file']))
    #current_page = reader.pages[int(row['page'])]
    #writer.add_page(current_page)

    
for i, (index, row) in enumerate(timex_df.iterrows()):
    
    if i % 10 == 0:
        print("CHECKPOINT")
        df.to_csv(file)
            
    if i < 3000:
        continue
    if i > STOP:
        print("STOPPED AT", i, index)
        break
    print("###########################################################################")
    print(i, index)
    if pd.isna(row['file']):
        context_window = row['Context']
    else:
        reader = pypdf.PdfReader('./data/helsearkiv/journal/' + str(row['file']))
        current_page = reader.pages[int(row['page'])]
        context_window = current_page.extract_text()
        
        writer = pypdf.PdfWriter()
    print(f"\nText: {row['Text']}, TIMEX: {row['DCT']}")
    print(f"Context: {context_window.replace(row['Text'], '\n\n <TAG>' + row['Text'] + '</TAG> \n\n')}")
    
    user_input = input("Enter 'c' to keep, '1' for BEFORE, '2' for OVERLAP, '3' for AFTER: ").strip().lower()
    
    if user_input == '1':
        df.at[index, 'DCT'] = "BEFORE"
    elif user_input == '2':
        df.at[index, 'DCT'] = "OVERLAP"
    elif user_input == '3':
        df.at[index, 'DCT'] = "AFTER"  # Set TIMEX value to NaN
    elif user_input == '4':
        df.at[index, 'DCT'] = np.nan


# Save the modified file
df.to_csv(file)

print("Processing complete!")
