import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime


file = "./data/helsearkiv/batch/dtr/1-final.csv"   
df = pd.read_csv(file)

STOP = len(df)
df = df.sort_values(by="prob", ascending=True)  # Sort in ascending order


print("LEN:", len(df))

for i, (index, row) in enumerate(df.iterrows()):
    if i < 0:
        continue
    if i > STOP:
        print("STOPPED AT", i, index)
        break
    print("###########################################################################")
    print(i, index)
    
    print(f"\nFROM Text: {row['FROM']}, TO Text: {row['TO']}, RELATION: {row['RELATION']}")
    
    if row['TO'] not in row['FROM_CONTEXT']:
        df.at[index, 'RELATION'] = np.nan
    

    print(f"Context: {row['FROM_CONTEXT'].replace(row['FROM'], '<FROM-TAG>' + row['FROM'] + '</FROM-TAG>').replace(row['TO'], '<TO-TAG>' + row['TO'] + '</TO-TAG>')}")
    
    user_input = input("Enter 'c' to keep, '1' for BEFORE, '2' for OVERLAP, '3' for None: ").strip().lower()
    
    if user_input == '1':
        df.at[index, 'RELATION'] = "BEFORE"
    elif user_input == '2':
        df.at[index, 'RELATION'] = "OVERLAP"
    elif user_input == '3':
        df.at[index, 'RELATION'] = np.nan # Set TIMEX value to NaN
        
    if i % 30 == 0:
        print("CHECKPOINT")
        df.to_csv(file, index=True)


# Save the modified file
df.iloc[:STOP].to_csv(file, index=True)

print("Processing complete!")
