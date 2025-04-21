import os
import pandas as pd
from preprocess.dataset import DatasetManager
from structure.enum import Dataset
from textmining.ner.lexicon import Lexicon
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#folder_path = "./data/helsearkiv/annotated/entity/"

#entity_files = [
    #folder_path + f
    #for f in os.listdir(folder_path)
    #if os.path.isfile(os.path.join(folder_path, f))
#]

#folder_path = "./data/helsearkiv/annotated/relation/"

#relation_files = [
    #folder_path + f
    #for f in os.listdir(folder_path)
    #if os.path.isfile(os.path.join(folder_path, f))
#]

#folder_path = "./data/helsearkiv/batch/ner/final/"

#batch_files = [
    #folder_path + f
    #for f in os.listdir(folder_path)
    #if os.path.isfile(os.path.join(folder_path, f)) and f.split("-")[0] in ['b1', 'b2', 'b3']
#]

#entity_files.extend(batch_files)

#manager = DatasetManager(entity_files, relation_files)
#raw_dataset = manager.get(Dataset.NER)

#raw_dataset["MedicalEntity"] = raw_dataset["MedicalEntity"].fillna("O")

#dataset = []
#tags = set()
#for _, row in raw_dataset.iterrows():
    #dataset.append(
        #(row["Text"], row["MedicalEntity"])
    #)  # Add (row[1], row[2]) tuple to list
    #tags.add(row["MedicalEntity"])  # Add row[2] to the set
    
#train, test = train_test_split(
    #dataset,
    #train_size=0.8,
    #random_state=42,
#)

#train, val = train_test_split(train, train_size=0.8, random_state=42)

#sentences = []
#target = []

#dataset = test

#for term in dataset:
    #splitted_term = str(term[0]).split()
    #for word in splitted_term:
        #target.append(term[1])
    #sentences.extend(splitted_term)
    
#assert len(sentences) == len(target), f"sentences: {len(sentences)}, target: {len(target)}"

#sentences = " ".join(sentences)

#lex = Lexicon()

#result = lex.run(sentences)

#result = [
    #res.replace("SYMPTOM", "CONDITION").replace("EVENT", "TREATMENT")
    #for res in result
#]

#print("LOCAL")
#print(classification_report(target, result, labels=["CONDITION", "TREATMENT"]))

### TEST set 
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
raw_dataset = manager.get(Dataset.NER)

raw_dataset["MedicalEntity"] = raw_dataset["MedicalEntity"].fillna("O")

sentences = []
dataset = []
target = []
tags = set()
for _, row in raw_dataset.iterrows():
    dataset.append(
        (row["Text"], row["MedicalEntity"])
    )  # Add (row[1], row[2]) tuple to list
    tags.add(row["MedicalEntity"])  # Add row[2] to the set

for term in dataset:
    splitted_term = str(term[0]).split()
    for word in splitted_term:
        target.append(term[1])
    sentences.extend(splitted_term)
    
assert len(sentences) == len(target), f"sentences: {len(sentences)}, target: {len(target)}"

sentences = " ".join(sentences)

lex = Lexicon()

result = lex.run(sentences)

result = [
    res.replace("SYMPTOM", "CONDITION").replace("EVENT", "TREATMENT")
    for res in result
]

print("GLOBAL")
print(classification_report(target, result, labels=["CONDITION", "TREATMENT"]))