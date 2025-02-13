import os
import pypdf
import pandas as pd
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from util import compute_mnlp
from structure.enum import Dataset

BATCH = 1

file = "./scripts/active-learning/config/dtr.ini"
save_directory = "./models/dtr/a-bilstm"

print("##### Start active learning for DTR... ######")

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
tre = TRExtract(
        config_file=file,
        manager=manager,
        save_directory=save_directory,
        task=Dataset.DTR
)
model = tre.get_model()

files = []
raw_files = os.listdir('./data/helsearkiv/journal')
annotated_files = os.listdir('./data/helsearkiv/annotated')
annotated_files = [file.replace('.pdf', '') for file in annotated_files]

files = [file for file in raw_files if file.replace('.pdf', '' not in annotated_files)]

al_data = []

print("##### Calculating MNLP ... ")
for i, doc in enumerate(files):
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
    for j, page in enumerate(reader.pages):
        prob = compute_mnlp(model)
        
        al_data.append({
            'file': doc,
            'page': j,
            'prob': prob
        })
        
        
sorted_data = sorted(al_data, key=lambda x: x['prob'])

writer = pypdf.PdfWriter()
count = 0
csv_file = []
for page in sorted_data:
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + page['file'])
    writer.add_page(reader.pages[page['page']])
    csv_file.append(page)
    count += 1
    
    if count % 50 == 0:
        os.mkdir(f'./data/helsearkiv/batch/tre/{BATCH}')
        with open(f"./data/helsearkiv/batch/tre/{BATCH}/{count // 50}.pdf", "wb") as file:
            writer.write(file)
        
        writer = pypdf.PdfWriter()
        

df = pd.DataFrame(csv_file)
df.to_csv(f'./data/helsearkiv/batch/tre/{BATCH}')
