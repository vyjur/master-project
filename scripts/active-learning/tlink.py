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
    
entities = pd.read_csv("./data/helsearkiv/batch/ner/all-local/all.csv")
entities = entities[(entities["MedicalEntity"].notna()) & (entities["MedicalEntity"] != "O")]
timex = pd.read_csv(f"./data/helsearkiv/batch/tee/{1}.csv")
timex = timex[timex["TIMEX"].notna()] 

all_entities = pd.concat([entities, timex])

batch_path = './data/helsearkiv/batch/tlink/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv') and 'final' in f]

batch_entities = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
batch_entities = pd.concat(batch_entities, ignore_index=True)  # Combine all into one DataFrame

entities = pd.read_csv(all_entities)

filtered_entities = entities.merge(batch_entities, on=["FROM", "FROM_Id", "TO", "TO_Id", "file", "page"], how="left", indicator=True)
print(len(entities), len(filtered_entities))
dataset = filtered_entities[filtered_entities["_merge"] == "left_only"].drop(columns=["_merge"])


grouped_df = dataset.groupby(['file', 'page'])

al_data = []
for name, group in grouped_df:
    pairs = []
    metadata = []
    
    for i, e_i in group.iterrows():
        for j, e_j in group.iterrows():
            if i == j:
                continue
            
            if e_j['Text'] not in e_i['Context']:
                if random.random() < 0.7:
                    continue
                
            pairs.append((e_i, e_j))
            metadata.append((i, e_i, j, e_j))
            
            pred, prob = tlink.run(e_i, e_j)
            al_data.append({
                'FROM':e_i['Text'],
                'FROM_Id':e_i['Id'],
                'FROM_CONTEXT':e_i['Context'],
                'TO':e_j['Text'],
                'TO_Id': e_j['Id'],
                'TO_CONTEXT':e_j['Context'],
                'RELATION': pred,
                'file': i[0],
                'page': i[1],
                'prob': prob
            })
            
    if pairs:
        # Run batch processing
        results = tlink.batch_run(pairs)  # Assuming it returns a list of (pred, prob)
        
        # Append results efficiently
        for (i, e_i, j, e_j), (pred, prob) in zip(metadata, results):
            al_data.append({
                'FROM': e_i['Text'],
                'FROM_Id': e_i['Id'],
                'FROM_CONTEXT': e_i['Context'],
                'TO': e_j['Text'],
                'TO_Id': e_j['Id'],
                'TO_CONTEXT': e_j['Context'],
                'RELATION': pred,
                'file': i[0],
                'page': i[1],
                'prob': prob.cpu().item()
            })
al_df = pd.DataFrame(al_data)
al_df = al_df.sort_values('prob', ascending=True)
al_df.to_csv(f'./data/helsearkiv/batch/tlink/{BATCH}.csv')