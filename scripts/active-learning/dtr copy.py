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
from util import compute_mnlp
from types import SimpleNamespace
from datetime import datetime

BATCH = 1
SIZE = 50

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

file = "./scripts/active-learning/config/tre-dtr.ini"
save_directory = "./models/tre/dtr/model/b-bert"
dtr = TRExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
    task=Dataset.DTR

)

files = []
folder_path = "./data/helsearkiv/batch/base_ner/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/batch/tee/"

timex_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
entity_files.extend(timex_files)

new_manager = DatasetManager(entity_files, [])

batch_path = "./data/helsearkiv/batch/dtr/"
csv_files = [f for f in os.listdir(batch_path) if f.endswith(".csv")]

# Read and merge all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if df_list:
    batch_df = pd.concat(df_list)
else:
    batch_df = pd.DataFrame(
        columns=[
            "Text",
            "Id",
            "MedicalEntity",
            "DCT",
            "TIMEX",
            "Context",
            "sentence-id",
            "Relation",
            "file",
            "page",
        ]
    )

dataset = new_manager.get(Dataset.DTR)

dataset['prob'] = 0

for i, row in dataset.iterrows():
    
    if i in batch_df.index:
        continue
    
    e = {'value': row['Text'], 'context': row['Context']}
    output = dtr.run(SimpleNamespace(**e))
    dataset[i, 'DCT'] = output[0]
    dataset[i, 'prob'] = output[1].item()

dataset = dataset.sort_values('prob', ascending=True)
dataset.to_csv(f'./data/helsearkiv/batch/dtr/{BATCH}.csv')
