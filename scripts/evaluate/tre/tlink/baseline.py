import os
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


manager = DatasetManager(entity_files, relation_files)
dataset_ner = manager.get(Dataset.NER)
dataset_ner = dataset_ner[
    dataset_ner["MedicalEntity"].notna() | dataset_ner["TIMEX"].notna()
].reset_index()
dataset_tre = manager.get(Dataset.TLINK)

dataset = []

for i, rel in dataset_tre.iterrows():

    e_i = dataset_ner[dataset_ner["Id"] == rel["FROM_Id"]]
    e_j = dataset_ner[dataset_ner["Id"] == rel["TO_Id"]]

    if len(e_i) > 0 and len(e_j) > 0:
        e_i = e_i.iloc[0]
        e_j = e_j.iloc[0]
    else:
        continue
    
    cat = TRExtract.classify_tlink(e_i, e_j)

    dataset.append({
        "e_i": e_i,
        "e_j": e_j,
        "cat": cat,
        "relation": rel['RELATION']
    })

train, test = train_test_split(
    dataset,
    train_size=0.8,
    random_state=42,
)

train, val = train_test_split(train, train_size=0.8, random_state=42)

pred = []
target = [i['relation'] for i in test if i['cat']]

for row in test:
    if row['cat'] == SENTENCE.INTRA:
        pred.append(baseline.run(row['e_i'], row['e_j']))

print(classification_report(target, pred, labels=list(tags)))
