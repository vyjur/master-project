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
import random

BATCH = 1
tee_start = 1

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

file = "./scripts/active-learning/config/tre-tlink.ini"
save_directory = "./models/tre/tlink/model/b-bert"
tlink = TRExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
    task=Dataset.TLINK
)

batch_path = './data/helsearkiv/batch/tlink/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv')]

# TODO: how can we do this
# Read and merge all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if df_list:
    batch_df = pd.concat(df_list, ignore_index=True)
else:
    batch_df = pd.DataFrame(columns=[
        "FROM",
        "FROM_Id",
        "FROM_CONTEXT",
        "TO",
        "TO_Id",
        "TO_CONTEXT",
        "RELATION",
        "file",
        "page",
        "prob"
    ])
    
batch_path = './data/helsearkiv/batch/ner/final/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv')]
df_list = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
entities = pd.concat(df_list, ignore_index=True)

#entities = pd.read_csv("./data/helsearkiv/batch/ner/all-al/all.csv")
entities = entities[(entities["MedicalEntity"].notna()) & (entities["MedicalEntity"] != "O")]

entities['file_id'] = entities['file'].str.split('_').str[1].str.strip()

# Filter where file_id is NOT in patients_df['journalidentifikator']
patients_df = pd.read_csv('./data/helsearkiv/patients.csv')
entities = entities[~entities['file_id'].isin(patients_df['journalidentifikator'])]

# Optionally drop the helper column
entities = entities.drop(columns='file_id')

timex = pd.read_csv(f"./data/helsearkiv/batch/tee/{1}.csv")
timex = timex[timex["TIMEX"].notna()] 

all_entities = pd.concat([entities, timex])

batch_path = './data/helsearkiv/batch/tlink/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv') and 'final' in f]

df_list = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if df_list:
    batch_df = pd.concat(df_list, ignore_index=True)
else:
    batch_df = pd.DataFrame(columns=[
        "FROM",
        "FROM_Id",
        "FROM_CONTEXT",
        "TO",
        "TO_Id",
        "TO_CONTEXT",
        "RELATION",
        "file",
        "page",
        "prob"
    ])



dataset = all_entities

BATCH_SIZE = 128

grouped_df = dataset.groupby(['file', 'page'])

al_data = []
for name, group in grouped_df:
    pairs = []
    metadata = []
    
    for i, e_i in group.iterrows():
        for j, e_j in group.iterrows():
            if i == j:
                continue
            
            if str(e_j['Text']) not in e_i['Context']:
                if random.random() < 0.7:
                    continue
                
            pairs.append((e_i, e_j))
            metadata.append((i, e_i, j, e_j))
            
    if pairs:
        
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i:i + BATCH_SIZE]
            results = tlink.batch_run(batch)  

            for (i, e_i, j, e_j), (pred, prob) in zip(metadata, zip(results[0], results[1])):

                al_data.append({
                    'FROM': e_i['Text'],
                    'FROM_Id': e_i['Id'],
                    'FROM_CONTEXT': e_i['Context'],
                    'TO': e_j['Text'],
                    'TO_Id': e_j['Id'],
                    'TO_CONTEXT': e_j['Context'],
                    'RELATION': pred,
                    'file': e_i['file'],
                    'page': e_i['page'],
                    'prob': prob.cpu().item()
                })
al_df = pd.DataFrame(al_data)


filtered_entities = al_df.merge(batch_df, on=["FROM", "FROM_Id", "TO", "TO_Id", "file", "page"], how="left", indicator=True)
print(len(al_df), len(filtered_entities))
filtered_entities = filtered_entities.sort_values('prob', ascending=True)
filtered_entities.to_csv(f'./data/helsearkiv/batch/tlink/{BATCH}.csv')