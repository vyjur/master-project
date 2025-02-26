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

files = []
folder_path = "./data/helsearkiv/batch/base_ner/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/batch/tee/"

tee_entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

entity_files.extend(tee_entity_files)

new_manager = DatasetManager(entity_files, [])

dataset = new_manager.get(Dataset.TLINK)

grouped_df = dataset.groupby(['file', 'page'])

al_data = []
for name, group in grouped_df:
    for i, e_i in group.iterrows():
        for j, e_j in group.iterrows():
            if i == j:
                continue
            
            if e_j['Text'] not in e_i['Context']:
                if random.random() < 0.5:
                    continue
            pred, prob = tlink.run(e_i, e_j)
            al_data.append({
                'FROM':e_i['Text'],
                'FROM_Id':e_i['Id'],
                'FROM_CONTEXT':e_i['Context'],
                'TO':e_j['Text'],
                'TO_Id': e_j['Id'],
                'TO_CONTEXT':e_j['Context'],
                'RELATION': pred,
                'file': name,
                'page': name,
                'prob': prob
            })
            
al_df = pd.DataFrame(al_data)
al_df = al_df.sort_values('prob', ascending=True)
al_df.to_csv(f'./data/helsearkiv/batch/tlink/{BATCH}.csv')