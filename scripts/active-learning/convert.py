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

init_time = datetime.now()

BATCH = 102

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

print("##### Start active learning for NER... ######")

file = "./scripts/active-learning/config/ner.ini"
save_directory = "./models/ner/model/b-bert"
ner = NERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

file = "./scripts/active-learning/config/tee.ini"
save_directory = "./models/tee/model/b-bert"
tee = TEExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory
)

file = "./scripts/active-learning/config/tre-dtr.ini"
save_directory = "./models/tre/dtr/model/b-bert"
dtr = TRExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
    task=Dataset.DTR

)

file = "./scripts/active-learning/config/tre-tlink.ini"
save_directory = "./models/tre/tlink/model/b-bert"
tlink = TRExtract(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
    task=Dataset.TLINK
)
preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)

model = ner.get_model()

os.mkdir(f'./data/helsearkiv/batch/ner/{BATCH}-local')

files = ["d1d07d02-8240-4a46-a007-85a69a06def4_0F978AAC-681F-4A61-9F82-0D00316D8FDC.pdf"]

al_data = []

patients_df = pd.read_csv('./data/helsearkiv/patients.csv')

new_time = datetime.now()
print("Init time: ", new_time-init_time)
init_time = new_time

print("##### Calculating MNLP ... ")
for i, doc in enumerate(files):
    if doc.split("_")[1].strip() not in patients_df['journalidentifikator'] or True:
        reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
        for j, page in enumerate(reader.pages[4:5]):
            pre_output = preprocess.run(page.extract_text())
            al_data.append({
                'file': doc,
                'page': j+4,
            })
        
        
sorted_data = al_data

writer = pypdf.PdfWriter()
count = 0
info_file = []

csv_file = []


print("### Performing NER with the current model")

merged_entities = []
merged_annots = []
merged_offsets = []


for page in sorted_data:
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
        
        tokens_expanded = [word for entity in merged_entities for word in entity.split()]
        
        contexts = []
        
        new_time = datetime.now()
        print("NER time:", new_time - init_time)
            
        ### TIMEX
        timex = []
        for i, ent in enumerate(merged_entities):
            context = None
            if merged_annots[i] != None and len(ent) >1:
                result = tee.run(ent)
                if len(result) < 1:
                    timex.append(None)
                else:
                    timex_output = result.iloc[0]['type']
                    if timex_output == "DATE":
                        context = manager.get_context_window(i, merged_entities, tokens_expanded)
                        e = {
                            'Text': ent,
                            'context': context
                        }
                        timex_output = tee.predict_sectime(e)
                    timex.append(timex_output[0])
            else:
                timex.append(None)
            contexts.append(None)
            
        new_time = datetime.now()
        print("TIMEX time:", new_time - init_time)

        ### DocTimeRel        
        dcts = []

        for i, ent in enumerate(merged_entities):
            if merged_annots[i] != 'O' or timex[i] is not None:
                context = manager.get_context_window(i, merged_entities, tokens_expanded)
                e = {'value': ent, 'context': context}
                dcts.append(dtr.run(SimpleNamespace(**e))[0])
                contexts[i] = context
            else:
                dcts.append(None)
                
        new_time = datetime.now()
        print("DTR time:", new_time - init_time)
                
with open(f"./data/helsearkiv/batch/ner/{BATCH}-local/{count // 36}.pdf", "wb") as file:
    writer.write(file)
    
df = pd.DataFrame({
            'Text': [word.replace("[UNK]", "").replace("[SEP]", "").replace("[CLS]", "").replace("PAD", "") for word in merged_entities],
            'Id': [f"{offset[0]}-{offset[1]}"for offset in merged_offsets],
            'MedicalEntity': merged_annots,
            'DCT': dcts,
            'TIMEX': timex,
            'Context': contexts,
            'sentence-id': '',
            'Relation': ''
        })
df.to_csv(f"./data/helsearkiv/batch/ner/{BATCH}-local/{count // 36}.csv")

df = pd.DataFrame(info_file)
df.to_csv(f'./data/helsearkiv/batch/ner/{BATCH}.csv')


print("### Active Learning run finished!")
print("#################################################")

print("### Converting to webanno.tsv style")

out_folder_path = f"./data/helsearkiv/batch/ner/{BATCH}-webanno/"
in_folder_path = f"./data/helsearkiv/batch/ner/{BATCH}-local/"

os.mkdir(out_folder_path)

entity_files = [
    (in_folder_path + f, f)
    for f in os.listdir(in_folder_path)
    if os.path.isfile(os.path.join(in_folder_path, f))
]

for i, (path, file) in enumerate(entity_files):
        
    if ".pdf" in file:
        continue
    
    content = textwrap.dedent("""\
        #FORMAT=WebAnno TSV 3.3
        #T_SP=org.dkpro.core.api.pdf.type.PdfPage|height|pageNumber|width
        #T_SP=webanno.custom.MedicalEntity|DCT|MedicalEntity|TIMEX
        #T_RL=webanno.custom.TLINK|TLINK|BT_webanno.custom.MedicalEntity


    """)
    
    df = pd.read_csv(path)
    offset = 0
    sentence_id = 1
    token_id = 1
    word_id = 1
    annot_id = 1
    count_id = 0
    with open(path.replace(in_folder_path, out_folder_path).replace(file, f"b{BATCH}-{file}"), 'w') as out:
        df['Text'] = df['Text'].fillna('')
        content += f"#Text={" ".join(df['Text'].astype(str).values).replace('[UNK]', '').replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '').replace('  ', ' ').strip()}\n"
        
        total_words = 0
        for j, row in df.iterrows():
            row['Text'] = str(row['Text'])
            words = row['Text'].split()
            df.at[j, 'sentence-id'] = f'{sentence_id}-{total_words + 1}'
            total_words += len(words)
            
        for j, row in df.iterrows():
            row['Text'] = str(row['Text'])
            words = row['Text'].split()
            temp = ''
            if len(words) > 1:
                temp = f"[{word_id}]"
                word_id += 1
                
            entity = row['MedicalEntity'] if row['MedicalEntity'] not in ['O', None] else '_'
            timex = row['TIMEX'] if row['TIMEX'] is not None else '_'
            dct = row['DCT'] if row['DCT'] is not None else '_'
    
            col = None
            if entity is not None:
                col = 'MedicalEntity'
            elif timex is not None:
                col = 'TIMEX'
                
            relations = []
            if col is not None:
                for k, f_row in df[~df['MedicalEntity'].isin(['O']) & df['MedicalEntity'].notna() & df['TIMEX'].notna()]:
                    if f_row['TEXT'] in row['Context']:
                        relation = tlink.run(row, f_row)[0]
                        if relation != "O":
                            relations.append((f_row['sentence-id'], relation))    

            for k, word in enumerate(words):
                word = word.replace('[UNK]', '').replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '')
                if word == '' or word is None or pd.isna(word) or word == 'nan':
                    continue
                
                rel_cat = ''
                rel_fk = ''
                if k == 0 and len(relations) > 0:
                    rel_cat = "|".join([rel[1] for rel in relations])
                    rel_fk = "|".join([rel[0] for rel in relations])
                count_id += 1
                content += f'{sentence_id}-{token_id}	{offset}-{offset + len(word)+1}\t{word}\t_\t_\t_\t{dct}{temp}\t{entity}{temp}\t{timex}{temp}\t_\t_\n' 
                offset += len(word) + 1
                token_id += 1
        
        out.write(content)