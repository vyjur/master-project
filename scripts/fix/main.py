import os
import pandas as pd
import numpy as np  # Import NumPy for NaN values
from datetime import datetime

folder_path = "./data/helsearkiv/annotated/entity/"

# Get all CSV files in the folder
entity_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")
]

print("LEN:", len(entity_files))

# Process each file
for index, file in enumerate(entity_files):
    print("###INDEX:", index)
    mod_time = os.path.getmtime(file)

    # Convert to a readable date format
    mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")
    
    df = pd.read_csv(file)
    
    # Filter rows where TIMEX is not NaN
    timex_df = df[~df['TIMEX'].isna()]
    
    for index, row in timex_df.iterrows():
        print("###########################################################################")
        print(f"\nText: {row['Text']}, TIMEX: {row['TIMEX']}")
        window_size = 10
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

    # Save the modified file
    df.to_csv(file, index=False)
    print(f"Updated file saved: {file}")

print("Processing complete!")
