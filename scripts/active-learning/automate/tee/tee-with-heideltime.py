import os
import re
import pypdf
import textwrap
import pandas as pd
from textmining.mer.setup import MERecognition
from textmining.tee.setup import TEExtract
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess
from structure.enum import Dataset
from util import compute_mnlp
from types import SimpleNamespace
from datetime import datetime

# 15: not all one doc only
# 16: tee2 all
# 17 one page
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

file = "./scripts/active-learning/config/tee.ini"
save_directory = "./models/tee/model/b-bert"
tee = TEExtract(config_file=file, manager=manager, save_directory=save_directory)

file = "./scripts/active-learning/config/ner.ini"
save_directory = "./models/ner/model/b-bert"
ner = MERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)


files = []
raw_files = os.listdir("./data/helsearkiv/journal")
annotated_files = os.listdir("./data/helsearkiv/annotated")
annotated_files = [file.replace(".pdf", "") for file in annotated_files]

batch_path = "./data/helsearkiv/batch/tee/"
csv_files = [f for f in os.listdir(batch_path) if f.endswith(".csv")]

# Read and merge all CSV files into one DataFrame
df_list = [pd.read_csv(os.path.join(batch_path, file)) for file in csv_files]
if df_list:
    batch_df = pd.concat(df_list, ignore_index=True)
else:
    batch_df = pd.DataFrame(
        columns=[
            "Text",
            "Id",
            "MedicalEntity",
            "DCT",
            "TIMEX",
            "Context",
            "sentence-id",
            "Relation",
            "file",
            "page",
        ]
    )

files = [file for file in raw_files]

al_data = []

patients_df = pd.read_csv("./data/helsearkiv/patients.csv")
info_file = []

print("##### Calculating MNLP ... ")
for i, doc in enumerate(files):
    if doc.split("_")[1].strip() not in patients_df["journalidentifikator"]:
        reader = pypdf.PdfReader("./data/helsearkiv/journal/" + doc)
        for j, page in enumerate(reader.pages):
            if not batch_df[(batch_df['file'] == doc) & (batch_df['page'] == page)].empty:
                continue
            
            try:
                text = page.extract_text()
                
                if len(text) > 0:
                    result = tee.run(page.extract_text(), True)
                    added = True
                else:
                    added = False
            except:
                added = False

            row = {
                'file': doc,
                'page': j,
                'added': added
            }
            
            if not added:
                continue
            info_file.append(row)
            entities = []
            for _, res in result.iterrows():
                if res["type"] == "DATE" and not 'ICD' in res['context'] and res["text"].strip() != '': 
                    entities.append(res)
                                
            tee_entities = [
                {"text": res["text"], "context": res["context"]} for res in entities
            ]
            
            timex_output = tee.batch_predict_sectime(tee_entities, True)
            
            for i, res in enumerate(entities):
                al_data.append(
                    {
                        "Text": res["text"],
                        "Id": f"tee-{tee_start}",
                        "MedicalEntity": "O",
                        "DCT": "OVERLAP",
                        "TIMEX": timex_output[0][i],
                        "Context": res["context"],
                        "sentence-id": "",
                        "Relation": "",
                        "file": doc,
                        "page": j,
                        "prob": timex_output[1][i]
                    }
                )
                tee_start += 1

info_df = pd.DataFrame(info_file)
info_df.to_csv(f"./data/helsearkiv/batch/tee/info-{BATCH}.csv")
al_df = pd.DataFrame(al_data)
print("LENGTH:", len(al_data))
al_df = al_df.sort_values(by="prob",ascending=True)

al_df.to_csv(f"./data/helsearkiv/batch/tee/{BATCH}.csv")

