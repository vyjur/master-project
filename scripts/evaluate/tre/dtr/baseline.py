import os
import pandas as pd
from preprocess.dataset import DatasetManager
from structure.enum import Dataset, DocTimeRel
from textmining.tre.baseline_dtr import Baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

folder_path = "./data/helsearkiv/annotated/entity-w-dct/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

# DTR batch
folder_path = "./data/helsearkiv/batch/dtr-w-dct/"

batch_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and "final" in f and f.split("-")[0] in ["1", "2"]
]

entity_files.extend(batch_files)

folder_path = "./data/helsearkiv/annotated/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

baseline = Baseline()
tags = [cat.name for cat in DocTimeRel]   


manager = DatasetManager(entity_files, relation_files)
local_dataset = manager.get(Dataset.DTR)

folder_path = "./data/helsearkiv/test_dataset/csv/entity-w-dct/"

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

test_manager = DatasetManager(entity_files, relation_files)
global_dataset = test_manager.get(Dataset.DTR)
global_dataset = global_dataset[global_dataset['DCT'] != 'BEFOREOVERLAP']


dataset = []

for _, row in local_dataset.iterrows():
    if "ICD" in row['Context']:
        continue
    sentences = sent_tokenize(row['Context'])
    for sentence in sentences:
        if row['Text'] in sentence:
            dataset.append((row['Text'], sentence , row['DCT']))

train, test = train_test_split(
    dataset,
    train_size=0.8,
    random_state=42,
)

train, val = train_test_split(train, train_size=0.8, random_state=42)

pred = []
target = [i[2] for i in test]

for row in test:
    pred.append(baseline.run(row[1]))
    
print("LOCAL")
print(classification_report(target, pred, labels=list(tags)))   

dataset = []

for _, row in global_dataset.iterrows():
    if "ICD" in row['Context']:
        continue
    sentences = sent_tokenize(row['Context'])
    for sentence in sentences:
        if row['Text'] in sentence:
            dataset.append((row['Text'], sentence , row['DCT']))

train, test = train_test_split(
    dataset,
    train_size=0.8,
    random_state=42,
)

train, val = train_test_split(train, train_size=0.8, random_state=42)

pred = []
target = [i[2] for i in test]

for row in test:
    pred.append(baseline.run(row[1]))
    
print("GLOBAL")
print(classification_report(target, pred, labels=list(tags)))   