import os
import pandas as pd
from preprocess.dataset import DatasetManager
from structure.enum import Dataset, TLINK, SENTENCE
from textmining.tre.baseline_tlink import Baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textmining.tre.setup import TRExtract

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize

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

baseline = Baseline()
tags = [cat.name for cat in TLINK]
tags.remove("BEFORE")


manager = DatasetManager(entity_files, relation_files)
dataset_ner = manager.get(Dataset.NER)
dataset_ner = dataset_ner[
    dataset_ner["MedicalEntity"].notna() | dataset_ner["TIMEX"].notna()
].reset_index()
dataset_tre = manager.get(Dataset.TLINK)
dataset_tee = manager.get(Dataset.TEE)

dataset = []

for i, rel in dataset_tre.iterrows():

    e_i = dataset_ner[dataset_ner["Id"] == rel["FROM_Id"]]
    e_j = dataset_ner[dataset_ner["Id"] == rel["TO_Id"]]

    if len(e_i) > 0 and len(e_j) > 0:
        e_i = e_i.iloc[0]
        e_j = e_j.iloc[0]
        
        if pd.isna(e_i['TIMEX']) and pd.isna(e_j['TIMEX']):
            continue
        
        if 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
            continue
    else:
        continue
    
    dataset.append({
        "e_i": e_i,
        "e_j": e_j,
        "relation": rel['RELATION']
    })
    
for i, e_i in dataset_ner.iterrows():
    for j, e_j in dataset_tee.iterrows():
        if 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
            continue
        
        if str(e_i['Text']) not in e_j['Context'] and str(e_j['Text']) not in e_i['Context']:
            continue
        
        relations = dataset_tre[
            (dataset_tre["FROM_Id"] == e_i['Id'])
            & (dataset_tre["TO_Id"] == e_j['Id'])
        ]

        if len(relations) > 0:
            continue
        else:
            relation = "O"
        e_j['MedicalEntity'] = None
        e_i['TIMEX'] = None
        dataset.append({
            "e_i": e_i
            ,
            "e_j": e_j,
            "relation": relation
        })

train, test = train_test_split(
    dataset,
    train_size=0.8,
    random_state=42,
)

train, val = train_test_split(train, train_size=0.8, random_state=42)

pred = []
target = [i['relation'] if i['relation'] in ["O", "OVERLAP"] else "O" for i in test]

for i, row in enumerate(test):
    
    result = baseline.run(row['e_i'], row['e_j'])
    pred.append(result)
    
    if result != target[i] and i % 5:
        print("######")
        
        result = baseline.run(row['e_i'], row['e_j'], True)
        print(target[i], result)
        print("++++++")
        print(f"1: {row["e_i"]['Text']} 2: {row["e_j"]['Text']}")
        print(row["e_i"]['Context'].replace(row["e_i"]['Text'], f"<TAG>{row["e_i"]['Text']}</TAG").replace(row["e_j"]['Text'], f"<TAG>{row["e_j"]['Text']}</TAG"))

print("LOCAL")
print(classification_report(target, pred, labels=list(tags)))


folder_path = "./data/helsearkiv/test_dataset/csv/entity/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/test_dataset/csv/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

manager = DatasetManager(entity_files, relation_files)
dataset_ner = manager.get(Dataset.NER)
dataset_ner = dataset_ner[
    dataset_ner["MedicalEntity"].notna() | dataset_ner["TIMEX"].notna()
].reset_index()
dataset_tre = manager.get(Dataset.TLINK)
print(len(dataset_tre))
dataset_tee = manager.get(Dataset.TEE)

dataset = []

for i, rel in dataset_tre.iterrows():
        
    e_i = dataset_ner[dataset_ner["Id"] == rel["FROM_Id"]]
    e_j = dataset_ner[dataset_ner["Id"] == rel["TO_Id"]]
    
    if len(e_i) > 0 and len(e_j) > 0:
        e_i = e_i.iloc[0]
        e_j = e_j.iloc[0]
        
        if 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
            continue
    else:
        continue
    
    dataset.append({
        "e_i": e_i,
        "e_j": e_j,
        "relation": rel['RELATION']
    })
    
for i, e_i in dataset_ner.iterrows():
    for j, e_j in dataset_tee.iterrows():
        if 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
            continue
        
        if str(e_i['Text']) not in e_j['Context'] and str(e_j['Text']) not in e_i['Context']:
            continue
        
        relations = dataset_tre[
            (dataset_tre["FROM_Id"] == e_i['Id'])
            & (dataset_tre["TO_Id"] == e_j['Id'])
        ]

        if len(relations) > 0:
            continue
        else:
            relation = "O"

        e_j['MedicalEntity'] = None
        e_i['TIMEX'] = None
        dataset.append({
            "e_i": e_i
            ,
            "e_j": e_j,
            "relation": relation
        })
        
pred = []
target = [i['relation'] for i in dataset]

for i, row in enumerate(dataset):
    pred.append(baseline.run(row['e_i'], row['e_j']))

print("GLOBAL")
print(classification_report(target, pred, labels=list(tags)))

