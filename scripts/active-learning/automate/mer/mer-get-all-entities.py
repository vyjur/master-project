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
from collections import Counter
import itertools

def most_common_element(lst):
    counts = Counter(lst)
    max_freq = max(counts.values())  # Find the highest frequency
    candidates = [k for k, v in counts.items() if v == max_freq]  # Get all elements with max frequency
    
    # If elements are comparable by length (like strings or lists), return the longest
    # Otherwise, return the numerically largest
    return max(candidates, key=lambda x: (len(str(x)), x))

BATCH = "all"
SIZE = 50
PAGES = 4300

os.mkdir(f'./data/helsearkiv/batch/ner/{BATCH}-local')

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

files = [file for file in raw_files]

al_data = []

patients_df = pd.read_csv('./data/helsearkiv/patients.csv')

model = ner.get_model()

count = 0
info_file = []

csv_file = []


print("### Performing NER with the current model")

merged_entities = []
merged_annots = []
merged_offsets = []
merged_files = []
merged_pages = []

for index, file in enumerate(files):
    if file.split("_")[1].strip() not in patients_df['journalidentifikator']:
        continue
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + file)
    info_file.append(file)
    
    for page in reader.pages:
        count += 1
        text = page.extract_text()
    
        cleaned = re.sub(r"[\"'“”‘’]", "", text)
        cleaned = cleaned.replace("Ä", "Æ")
        cleaned = cleaned.replace("Ö", "Ø")
        cleaned = cleaned.replace("¢", "")
        cleaned = cleaned.replace("¥", "")
        cleaned = cleaned.replace("™", "")
        cleaned = cleaned.replace("®", "")

        preoutput = preprocess.run(cleaned)
        output = ner.run(preoutput)
    
        for i, pre in enumerate(preoutput):
            decoded = preprocess.decode(pre.ids)
            cleaned = decoded.replace("[UNK]", "").replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").replace("[]", "")
            if len(cleaned) < 10:
                continue
            words = decoded.split(' ')

            word_count = 0
            curr = ''
            annot = []
            offsets = []
            cats = []
            
            start = True
            
            # Get the offset for each word from words that were split
            for j, cat in enumerate(output[i]):
                token = preprocess.decode(pre.ids[j])
                
                curr += token
                
                if start:
                    offsets.append(pre.offsets[j])  
                    start = False
                
                offsets[-1] = (offsets[-1][0], pre.offsets[j][1])
                cats.append(cat)

                if curr.strip() == words[word_count].strip():
                    cat = most_common_element(cats)
                    curr = ''
                    word_count += 1
                    
                    annot.append(cat)
                    cats = []
                    start = True
        
            assert len(words) == len(annot) == len(offsets), f"Word count:{word_count}, LEN words: {len(words)}, offset/annot: {len(offsets)}/{len(annot)}, \n words:{words} \n tokens: {token} \n curr: {curr}, word: {words[word_count]}, page: {page['page']}, file: {page['file']}"      
                
            # Merge different words together
            for j, word in enumerate(words):
                if word.strip() != '':
                    if annot[j] == 'O':
                        merged_entities.append(word)
                        merged_annots.append(annot[j])
                        merged_offsets.append(offsets[j])
                        
                        merged_files.append(page['file'])
                        merged_pages.append(page['page'])
                    elif annot[j].startswith('B-'):
                        merged_entities.append(word)
                        merged_annots.append(annot[j].replace('B-', ''))
                        merged_offsets.append(offsets[j])
                        
                        merged_files.append(page['file'])
                        merged_pages.append(page['page'])
                    else:
                        if len(merged_entities) > 0 and merged_annots[-1] == annot[j].replace('I-', '') and merged_files[-1] == page['file'] and merged_pages[-1] == page['page']:
                            merged_entities[-1] += ' ' + word
                            merged_offsets[-1] = (merged_offsets[-1][0], offsets[j][1])
                        else:
                            merged_entities.append(word)
                            merged_annots.append(annot[j].replace('I-', ''))
                            merged_offsets.append(offsets[j])
                            
                            merged_files.append(page['file'])
                            merged_pages.append(page['page'])
                        
if len(merged_entities) > 0:
    
    df = pd.DataFrame({
        'Text': [word.replace("[UNK]", "").replace("[SEP]", "").replace("[CLS]", "").replace("PAD", "").replace("[]", "") for word in merged_entities],
        'Id': [f"{offset[0]}-{offset[1]}"for offset in merged_offsets],
        'MedicalEntity': merged_annots,
        'DCT': [None for _ in range(len(merged_annots))],
        'TIMEX': [None for _ in range(len(merged_annots))],
        'Context': [None for _ in range(len(merged_annots))],
        'sentence-id': '',
        'Relation': '',
        'file':merged_files,
        'page':merged_pages
        })
    df.to_csv(f"./data/helsearkiv/batch/ner/{BATCH}-local/{BATCH}.csv")
df = pd.DataFrame(info_file)
df.to_csv(f'./data/helsearkiv/batch/ner/{BATCH}.csv')


print("#################################################")