import os
import re
import pypdf
import textwrap
import pandas as pd
from textmining.ner.setup import NERecognition
from textmining.tee.setup import TEExtract
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess
from structure.enum import Dataset
from types import SimpleNamespace
from datetime import datetime

BATCH = 1
SIZE = 50

folder_path = "./data/helsearkiv/annotated/entity-without-timex/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

for j, file in enumerate(entity_files):
    df = pd.read_csv(file)
    if j < 98: continue
    
    print("FILE:", j, file)
    
    for index, row in df.iterrows():
        if pd.isna(row['MedicalEntity']) or "ICD" in row['Text']:
            continue
        
        if j<= 98 and index < 1245:
            continue
        
        print(index, "########")
        print(row['Text'])
        
        user_input = input("Enter 'c' to keep, '1' for BEFORE, '2' for OVERLAP, '3' for AFTER: ").strip().lower()

        if user_input == '1':
            df.at[index, 'DCT'] = "BEFORE"
        elif user_input == '2':
            df.at[index, 'DCT'] = "OVERLAP"
        elif user_input == '3':
            df.at[index, 'DCT'] = "AFTER"  # Set TIMEX value to NaN

    print("SAVED!")
    df.to_csv(file)