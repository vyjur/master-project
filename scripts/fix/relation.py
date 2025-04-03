import os
import random
import pandas as pd
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "tlink"

folder = f"./scripts/train/tre/{TRE_TYPE}/config/model/"
configs = os.listdir(folder)

folder_path = "./data/helsearkiv/annotated/entity-without-timex/"

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

manager = DatasetManager(entity_files, relation_files, window_size=512)

dataset_tre = manager.get(Dataset.TLINK)
dataset_ner = manager.get(Dataset.NER)
dataset_ner = dataset_ner[dataset_ner['MedicalEntity'].notna()].reset_index()
dataset_tee = manager.get(Dataset.TEE)
full_dataset = pd.concat([dataset_ner, dataset_tee])

dataset = []

for i, e_i in full_dataset.iterrows():
    for j, e_j in full_dataset.iterrows():
        if i == j or 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
            continue
        
        if str(e_i['Text']) not in e_j['Context'] and str(e_j['Text']) not in e_i['Context']:
            continue
        
        if pd.notna(e_i['TIMEX']) and pd.notna(e_j['TIMEX']):
            continue
        
        if e_i['DCT'] != e_j['DCT']:
            continue

        relations = dataset_tre[
            (dataset_tre["FROM_Id"] == e_i['Id'])
            & (dataset_tre["TO_Id"] == e_j['Id'])
        ]

        if len(relations) > 0:
            continue
        else:
            relation = "O"

            # TODO: downsample majority class: 
            if random.random() < 0.7:
                continue
        
        relation_pair = {
            "FROM":e_i['Text'],
            "FROM_Id":e_i['Id'],
            "FROM_CONTEXT":e_i['Context'],
            "TO": e_j['Text'],
            "TO_Id":e_j['Id'],
            "TO_CONTEXT":e_j['Context'],
            "RELATION": relation
        }
        dataset.append(relation_pair)
        
df = pd.DataFrame(dataset)
df.to_csv('./data/helsearkiv/batch/tlink/0-1-final.csv')

for i, rel in dataset_tre.iterrows():
    if 'ICD' in rel['FROM_CONTEXT']:
        continue
    if rel['FROM_CONTEXT'].strip() == "" or rel['TO_CONTEXT'].strip() == "":
        e_i = dataset_ner[dataset_ner['Id'] == rel['FROM_Id']]
        e_j = dataset_ner[dataset_ner['Id'] == rel['TO_Id']]

        # Ensure values exist before assignment
        if not e_i.empty:
            dataset_tre.at[i, 'FROM_CONTEXT'] = e_i.iloc[0]['Context']
        if not e_j.empty:
            dataset_tre.at[i, 'TO_CONTEXT'] = e_j.iloc[0]['Context']
            
df.to_csv('./data/helsearkiv/batch/tlink/0-2-final.csv')
