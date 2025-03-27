import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime


file = "./data/helsearkiv/batch/dtr/1.csv"   
df = pd.read_csv(file)

STOP = len(df)
df = df.sort_values(by="prob", ascending=True)  # Sort in ascending order

# Filter rows where TIMEX is not NaN
timex_df = df[~df['TIMEX'].isna()]

print("LEN:", len(timex_df))

for i, (index, row) in enumerate(timex_df.iterrows()):
    if i < 0:
        continue
    if i > STOP:
        print("STOPPED AT", i, index)
        break
    print("###########################################################################")
    print(i, index)
    
    if row['Text'].replace(" ", "").isalpha():
        df.at[index, 'TIMEX'] = "DATE"
        continue

    print(f"\nText: {row['Text']}, TIMEX: {row['DCT']}")
    window_size = 500
    text_start = row["Context"].find(row["Text"])  # Find start position of the text

    if text_start != -1:  # Ensure text is found
        start = max(0, text_start - window_size // 2)
        end = min(len(row["Context"]), start + window_size)
        context_window = row["Context"][start:end]
    else:
        context_window = row["Context"][:window_size]  # Fallback if not found

    print(f"Context: {context_window.replace(row['Text'], '<TAG>' + row['Text'] + '</TAG>')}")
    
    user_input = input("Enter 'c' to keep, '1' for BEFORE, '2' for OVERLAP, '3' for AFTER: ").strip().lower()
    
    if user_input == '1':
        df.at[index, 'DCT'] = "BEFORE"
    elif user_input == '2':
        df.at[index, 'DCT'] = "OVERLAP"
    elif user_input == '3':
        df.at[index, 'DCT'] = "AFTER"  # Set TIMEX value to NaN
        
    if i % 30 == 0:
        print("CHECKPOINT")
        df.to_csv(file, index=True)


# Save the modified file
timex_df.iloc[:STOP].to_csv(file, index=True)

print("Processing complete!")
