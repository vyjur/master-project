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

BATCH = 1
SIZE = 50

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

manager = DatasetManager(entity_files, relation_files)

file = "./scripts/active-learning/config/tre-dtr.ini"
save_directory = "./models/tre/dtr/model/b-bert"
dtr = TRExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
    task=Dataset.DTR
)

# MEDICAL ENTITIES dont need the ones in entities because they alr trained
batch_path = "./data/helsearkiv/batch/ner/final/"
csv_files = [f for f in os.listdir(batch_path) if f.endswith(".csv")]
entities = pd.concat([pd.read_csv(os.path.join(batch_path, file)) for file in csv_files])
entities = entities[entities["MedicalEntity"].notna()]

# TIMEX
timex = pd.read_csv(f"./data/helsearkiv/batch/tee/{1}.csv")
timex = timex[timex["TIMEX"].notna()] 

all_entities = pd.concat([entities, timex])

batch_path = './data/helsearkiv/batch/dtr/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv') and 'final' in f]

batch_entities = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if batch_entities:  # Check if the list is not empty
    batch_entities = pd.concat(batch_entities, ignore_index=True)  # Combine all into one DataFrame
else:
    batch_entities = pd.DataFrame(columns=["page", "file", "Text", "Context"])  # Empty DataFrame with expected columns

filtered_entities = all_entities.merge(batch_entities, on=["page", "file", "Text"], how="left", indicator=True)
print(len(all_entities), len(filtered_entities))
dataset = filtered_entities[filtered_entities["_merge"] == "left_only"].drop(columns=["_merge"])

print(dataset['Context_x'])
raise ValueError

BATCH_SIZE = 64
batch_inputs = [
    SimpleNamespace(value=str(row['Text']), context=row['Context_x']) for _, row in dataset.iterrows()
]

# Prepare storage for results
dct_results = []
prob_results = []

# Process in batches
for i in range(0, len(batch_inputs), BATCH_SIZE):
    batch = batch_inputs[i:i + BATCH_SIZE]
    batch_outputs = dtr.batch_run(batch)

    # Collect results
    dct_results.extend(batch_outputs[0])
    prob_results.extend(output.cpu().item() for output in batch_outputs)

# Assign results to dataset
dataset['DCT'] = dct_results
dataset['prob'] = prob_results

dataset = dataset.sort_values('prob', ascending=True)
dataset.to_csv(f'./data/helsearkiv/batch/dtr/{BATCH}.csv')