import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime


file = "./data/helsearkiv/batch/tee/1-final.csv"   
df = pd.read_csv(file)

STOP = len(df)
df = df.sort_values(by="prob", ascending=False) 

# Filter rows where TIMEX is not NaN
timex_df = df[~df['TIMEX'].isna()]

print("LEN:", len(timex_df))

for i, (index, row) in enumerate(timex_df.iterrows()):
    if i < 1350:
        continue
    if i > STOP:
        print("STOPPED AT", i, index)
        break
    print("###########################################################################")
    print(i, len(timex_df))
    
    if row['Text'].replace(" ", "").isalpha():
        df.at[index, 'TIMEX'] = "DATE"
        continue

    print(f"\nText: {row['Text']}, TIMEX: {row['TIMEX']}")
    window_size = 100
    text_start = row["Context"].find(row["Text"])  # Find start position of the text

    if text_start != -1:  # Ensure text is found
        start = max(0, text_start - window_size // 2)
        end = min(len(row["Context"]), start + window_size)
        context_window = row["Context"][start:end]
    else:
        context_window = row["Context"][:window_size]  # Fallback if not found

    print(f"Context: {context_window.replace(row['Text'], '<TAG>' + row['Text'] + '</TAG>')}")
    
    user_input = input("Enter 'c' to keep, 'd' for DATE, 's' for DCT, 'n' to set NaN: ").strip().lower()
    
    if user_input == 'd':
        df.at[index, 'TIMEX'] = "DATE"
    elif user_input == 's':
        df.at[index, 'TIMEX'] = "DCT"
    elif user_input == 'n':
        df.at[index, 'TIMEX'] = np.nan  # Set TIMEX value to NaN
        
    if i % 10 == 0:
        print("CHECKPOINT")
        df.to_csv(file, index=True)


# Save the modified file
timex_df.iloc[:STOP].to_csv(file, index=True)

print("Processing complete!")
