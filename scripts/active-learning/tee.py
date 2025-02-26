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
tee = TEExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory
)

file = "./scripts/active-learning/config/ner.ini"
save_directory = "./models/ner/model/b-bert"
ner = NERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)


files = []
raw_files = os.listdir('./data/helsearkiv/journal')
annotated_files = os.listdir('./data/helsearkiv/annotated')
annotated_files = [file.replace('.pdf', '') for file in annotated_files]

#batch_path = './data/helsearkiv/batch/tee/'
#batch_files = []
#for filename in os.listdir(batch_path):
    #file_path = os.path.join(batch_path, filename)
    
    ## Read the file (assumes CSV, modify for other formats)
    #try:
        #df = pd.read_csv(file_path)  # Change to pd.read_excel() if needed
        ## Check if 'file' column exists
        #if "file" in df.columns:
            #batch_files.extend(df["file"].unique().dropna().tolist())  # Avoid NaNs
    #except Exception as e:
        #print(f"Skipping {filename}: {e}")

files = [file for file in raw_files if file.replace('.pdf', '') not in annotated_files]

al_data = []

patients_df = pd.read_csv('./data/helsearkiv/patients.csv')

print("##### Calculating MNLP ... ")
for i, doc in enumerate(files):
    if doc.split("_")[1].strip() not in patients_df['journalidentifikator']:
        reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
        for j, page in enumerate(reader.pages):
            try:
                result = tee.run(page.extract_text())
            except:
                continue
            for _, res in result.iterrows():
                if res['type'] == "DATE":
                    e = {
                        'Text': res['text'],
                        'context': res['context']
                    }
                    timex_output = tee.predict_sectime(e, True)
            
                    al_data.append({
                            'Text': res['text'],
                            'Id': f"tee-{tee_start}",
                            'MedicalEntity': 'O',
                            'DCT': 'OVERLAP',
                            'TIMEX': timex_output[0][0],
                            'Context': res['context'],
                            'sentence-id': '',
                            'Relation': '',
                            'file':doc,
                            'page':j
                        })
                tee_start += 1
                    
al_df = pd.DataFrame(al_data)
al_df = al_df.sort_values('prob')

al_df.to_csv(f'./data/helsearkiv/batch/tee/{BATCH}.csv')