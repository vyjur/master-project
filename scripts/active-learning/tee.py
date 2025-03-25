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

BATCH = 2
tee_start = 2

file = f"./data/helsearkiv/batch/tee/{BATCH}.csv"

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

conf_file = "./scripts/active-learning/config/tee.ini"
save_directory = "./models/tee/model/b-bert"
tee = TEExtract(config_file=conf_file, manager=manager, save_directory=save_directory)

al_data = []

batch_path = './data/helsearkiv/batch/tee/'
csv_files = [f for f in os.listdir(batch_path) if f.endswith('.csv') and 'final' in f]

batch_entities = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if batch_entities:  # Check if the list is not empty
    batch_entities = pd.concat(batch_entities, ignore_index=True)  # Combine all into one DataFrame
else:
    batch_entities = pd.DataFrame(columns=["page", "file", "Text", "Context"])  # Empty DataFrame with expected columns

all_entities = "./data/helsearkiv/batch/tee/1.csv"
entities = pd.read_csv(all_entities)

filtered_entities = entities.merge(batch_entities, on=["page", "file", "Text"], how="left", indicator=True)
print(len(entities), len(filtered_entities))
filtered_entities = filtered_entities[filtered_entities["_merge"] == "left_only"].drop(columns=["_merge"])

print(len(entities), len(filtered_entities))
print(filtered_entities.columns)

tee_entities = [
    {"text": res["Text"], "context": res["Context_x"]} for _, res in filtered_entities.iterrows()
]

# Define batch size
batch_size = 32  # Adjust based on performance needs
batch_predictions = []
batch_prob = []

# Process in batches
for i in range(0, len(tee_entities), batch_size):
    batch = tee_entities[i : i + batch_size]
    predictions = tee.batch_predict_sectime(batch, True)
    batch_predictions.extend(predictions[0])
    batch_prob.extend(predictions[1])
    
for i, (_, res) in enumerate(filtered_entities.iterrows()):
    al_data.append(
        {
            "Text": res["Text"],
            "Id": f"tee-{tee_start}",
            "MedicalEntity": "O",
            "DCT": "OVERLAP",
            "TIMEX": batch_predictions[i],
            "Context": res["Context_x"],
            "sentence-id": "",
            "Relation": "",
            "file": res['file'],
            "page": res['page'],
            "prob": batch_prob[i].cpu().item()
        }
    )
    tee_start += 1

al_df = pd.DataFrame(al_data)
print("LENGTH:", len(al_data))
al_df = al_df.sort_values(by="prob",ascending=True)

al_df.to_csv(f"./data/helsearkiv/batch/tee/{BATCH}-withprob.csv")

