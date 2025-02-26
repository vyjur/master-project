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

batch_path = './data/helsearkiv/batch/ner/'
batch_files = []
for filename in os.listdir(batch_path):
    file_path = os.path.join(batch_path, filename)
    
    # Read the file (assumes CSV, modify for other formats)
    try:
        df = pd.read_csv(file_path)  # Change to pd.read_excel() if needed
        # Check if 'file' column exists
        if "file" in df.columns:
            batch_files.extend(df["file"].unique().dropna().tolist())  # Avoid NaNs
    except Exception as e:
        print(f"Skipping {filename}: {e}")

files = [file for file in raw_files if file.replace('.pdf', '') not in annotated_files and file not in batch_files]

al_data = []

patients_df = pd.read_csv('./data/helsearkiv/patients.csv')

model = ner.get_model()

writer = pypdf.PdfWriter()
count = 0
info_file = []

csv_file = []


print("### Performing NER with the current model")

merged_entities = []
merged_annots = []
merged_offsets = []

for _, doc in enumerate(files):
    if doc.split("_")[1].strip() not in patients_df['journalidentifikator']:
        reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
        for k, page in enumerate(reader.pages):
            reader = pypdf.PdfReader('./data/helsearkiv/journal/' + page['file'])
            writer.add_page(reader.pages[page['page']])
            info_file.append(page)
            count += 1
            text = reader.pages[page['page']].extract_text()
            
            cleaned = re.sub(r"[\"'“”‘’]", "", text)
            cleaned = cleaned.replace("Ä", "Æ")
            cleaned = cleaned.replace("Ö", "Ø")
            cleaned = cleaned.replace("¢", "")

            preoutput = preprocess.run(cleaned)
            output = ner.run(preoutput)
        
            for i, pre in enumerate(preoutput):
                words = preprocess.decode(pre.ids).split(' ')

                word_count = 0
                curr = ''
                annot = []
                offsets = []
                
                start = True
                
                # Get the offset for each word from words that were split
                for j, cat in enumerate(output[i]):
                    token = preprocess.decode(pre.ids[j])
                    
                    curr += token
                    
                    if start:
                        annot.append(cat)
                        offsets.append(pre.offsets[j])  
                        start = False
                    
                    offsets[-1] = (offsets[-1][0], pre.offsets[j][1])

                    if curr.strip() == words[word_count].strip():
                        curr = ''
                        word_count += 1
                        start = True
            
                assert len(words) == len(annot) == len(offsets), f"Word count:{word_count}, LEN words: {len(words)}, offset/annot: {len(offsets)}/{len(annot)}, \n words:{words} \n tokens: {token} \n curr: {curr}, word: {words[word_count]}, page: {page['page']}, file: {page['file']}"      
                    
                # Merge different words together
                for j, word in enumerate(words):
                    if annot[j] == 'O':
                        merged_entities.append(word)
                        merged_annots.append(annot[j])
                        merged_offsets.append(offsets[j])
                    elif annot[j].startswith('B-'):
                        merged_entities.append(word)
                        merged_annots.append(annot[j].replace('B-', ''))
                        merged_offsets.append(offsets[j])
                    else:
                        if len(merged_entities) > 0 and merged_annots[-1] != 'O':
                            merged_entities[-1] += ' ' + word
                            merged_offsets[-1] = (merged_offsets[-1][0], offsets[j][1])
                        else:
                            merged_entities.append(word)
                            merged_annots.append(annot[j].replace('I-', ''))
                            merged_offsets.append(offsets[j])
                if count % SIZE == 0:
                    with open(f"./data/helsearkiv/batch/ner/{BATCH}-local/{count // SIZE}.pdf", "wb") as file:
                        writer.write(file)
                            
                        df = pd.DataFrame({
                                    'Text': [word.replace("[UNK]", "").replace("[SEP]", "").replace("[CLS]", "").replace("PAD", "") for word in merged_entities],
                                    'Id': [f"{offset[0]}-{offset[1]}"for offset in merged_offsets],
                                    'MedicalEntity': merged_annots,
                                    'DCT': [None for _ in range(len(merged_annots))],
                                    'TIMEX': [None for _ in range(len(merged_annots))],
                                    'Context': [None for _ in range(len(merged_annots))],
                                    'sentence-id': '',
                                    'Relation': '',
                                    "file": doc,
                                    "page": k
                                })
                        df.to_csv(f"./data/helsearkiv/batch/base_ner/{BATCH}-local/{count // SIZE}.csv")

df = pd.DataFrame(info_file)
df.to_csv(f'./data/helsearkiv/batch/base_ner/{BATCH}.csv')


print("### Active Learning run finished!")
print("#################################################")