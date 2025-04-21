import os
import re
import numpy as np
import textwrap
import pandas as pd
from textmining.mer.setup import MERecognition
from textmining.tee.setup import TEExtract
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess
from structure.enum import Dataset
from types import SimpleNamespace
from datetime import datetime

folder_path = "./data/helsearkiv/annotated/entity copy/"

# Get all CSV files in the folder
entity_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(".csv")
]

print("LEN:", len(entity_files))

BATCH = 0
file_count = 1

folder_path = "./data/helsearkiv/annotated/entity/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/annotated/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

manager = DatasetManager(entity_files, relation_files)

file = "./scripts/active-learning/config/tee.ini"
save_directory = "./models/tee/model/b-bert"
tee = TEExtract(config_file=file, manager=manager, save_directory=save_directory)

file = "./scripts/active-learning/config/ner.ini"
save_directory = "./models/ner/model/b-bert"
ner = MERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)

# Process each file
for index, file in enumerate(entity_files):
    print("###INDEX:", index)
    mod_time = os.path.getmtime(file)

    # Convert to a readable date format
    mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")
    
    df = pd.read_csv(file)
    
    # Filter rows where TIMEX is not NaN
    me_df = df[df["MedicalEntity"].notna()]

    timex_df = df[df["TIMEX"].notna() & df["TIMEX"].isin(["DATE", "DCT"])]
    
    keep_index = []
    for index, row in timex_df.iterrows():
        
        result = tee.run(row['Text'], sectime=False)
        if len(result) == 0:
            df.at[index, 'TIMEX'] = np.nan
        else:
            df.at[index, 'Text'] = result.iloc[0]['text']           
            df.at[index, 'Context'] = result.iloc[0]['context']

    # Save the modified file
    df.to_csv(f"./data/helsearkiv/batch/dtr/final/b{BATCH}-{file_count}", index=False)
    print(f"Updated file saved: {file}")
    file_count += 1

print("Processing complete!")
