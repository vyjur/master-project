import os
import re
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
ner = NERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)

texts = [
    'Inn 2.3.75 på poliklinikk',
    'Status presens 161279',
    'Status presens 1612792',

    'Status presens: 9.6.81',
    'Innl.dato: 08.12.93'
]

def convert_date_format_2(text):
    # Updated regex to match a valid 6-digit date (DDMMYY) and ensure no other numbers follow directly
    match = re.search(r"\b(\d{2}\d{2}\d{2})\b", text)
    
    if match:
        print("HEI", text)
        date_str = match.group(1)  # Extract the found date
        try:
            date_obj = datetime.strptime(date_str, "%d%m%y")  # Convert to date object
            formatted_date = date_obj.strftime("%d.%m.%y")  # Format as DD.MM.YY
            
            # Replace the original date in text with formatted date
            text = text.replace(date_str, formatted_date)
        except ValueError as e:
            print(f"Error parsing date: {e}")
    
    return text

for text in texts:
    result = tee.run(convert_date_format_2(text), sectime=True)
    print(result)
