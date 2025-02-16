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

for row in test:
    pred.append(baseline.run(row[1]))
    
print(classification_report(target, pred, labels=list(tags)))   
