import os
from preprocess.dataset import DatasetManager
from structure.enum import Dataset, DCT
from textmining.tee.baseline import Baseline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

folder_path = "./data/helsearkiv/annotated/entity/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/batch/tee/"

batch_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and "final" in f
]

entity_files.extend(batch_files)

folder_path = "./data/helsearkiv/annotated/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

baseline = Baseline()
tags = [DCT.DATE.name, DCT.DCT.name]   


manager = DatasetManager(entity_files, relation_files)
raw_dataset = manager.get(Dataset.TEE)

dataset = []

for _, row in raw_dataset.iterrows():
    sentences = sent_tokenize(row['Context'])
    if 'ICD' in row['Context']:
        continue
    for sentence in sentences:
        if row['Text'] in sentence:
            dataset.append((row['Text'], sentence , row['TIMEX']))

train, test = train_test_split(
    dataset,
    train_size=0.8,
    random_state=42,
)

train, val = train_test_split(train, train_size=0.8, random_state=42)

pred = []
target = [i[2] for i in test]

for i, row in enumerate(test):
    output = baseline.run(row)
    pred.append(output)
    
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

baseline = Baseline()
tags = [DCT.DATE.name, DCT.DCT.name]   


manager = DatasetManager(entity_files, relation_files)
raw_dataset = manager.get(Dataset.TEE)

dataset = []

for _, row in raw_dataset.iterrows():
    sentences = sent_tokenize(row['Context'])
    if 'ICD' in row['Context']:
        continue
    for sentence in sentences:
        if row['Text'] in sentence:
            dataset.append((row['Text'], sentence , row['TIMEX']))
            
pred = []
target = [i[2] for i in dataset]

for i, row in enumerate(dataset):
    output = baseline.run(row)
    pred.append(output)
    
print("GLOBAL")
print(classification_report(target, pred, labels=list(tags)))   
