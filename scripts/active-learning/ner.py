import os
import pypdf
import pandas as pd
from textmining.ner.setup import NERecognition
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess
from util import compute_mnlp

BATCH = 1

os.mkdir(f'./data/helsearkiv/batch/ner/{BATCH}')

file = "./scripts/active-learning/config/ner.ini"
save_directory = "./models/ner/b-bert"

print("##### Start active learning for NER... ######")

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
ner = NERecognition(
    config_file=file,
    manager=manager,
    save_directory=save_directory,
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length()
)

model = ner.get_model()

files = []
raw_files = os.listdir('./data/helsearkiv/journal')
annotated_files = os.listdir('./data/helsearkiv/annotated')
annotated_files = [file.replace('.pdf', '') for file in annotated_files]

files = [file for file in raw_files if file.replace('.pdf', '') not in annotated_files]

al_data = []

print("##### Calculating MNLP ... ")
for i, doc in enumerate(files):
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + doc)
    for j, page in enumerate(reader.pages):
        pre_output = preprocess.run(page.extract_text())
        prob = compute_mnlp(pre_output, model)
        
        al_data.append({
            'file': doc,
            'page': j,
            'prob': prob
        })
        
        
sorted_data = sorted(al_data, key=lambda x: x['prob'])

writer = pypdf.PdfWriter()
count = 0
info_file = []

csv_file = []


print("### Performing NER with the current model")

merged_entities = []
merged_annots = []
merged_offsets = []


for page in sorted_data[:1200]:
    reader = pypdf.PdfReader('./data/helsearkiv/journal/' + page['file'])
    writer.add_page(reader.pages[page['page']])
    info_file.append(page)
    count += 1
    
    preoutput = preprocess.run(reader.pages[page['page']].extract_text())
    output = ner.run(preoutput)
   
    for i, pre in enumerate(preoutput):
        words = preprocess.decode(pre.ids).split(' ')

        word_count = 0
        curr = ''
        annot = []
        offsets = []
        
        # Get the offset for each word from words that were split
        for j, cat in enumerate(output[i]):
            token = preprocess.decode(pre.ids[j])
            
            curr += token
            if curr.strip() == words[word_count].strip():
                curr = ''
                annot.append(cat)
                offsets.append(pre.offsets[j])
                word_count += 1
            
            if len(offsets) > 0:
                offsets[-1] = (offsets[-1][0], pre.offsets[j][1])
            else:
                annot.append(cat)
                offsets.append(pre.offsets[j])  
        
        assert len(words) == len(annot) == len(offsets), f"Word count:{word_count}, LEN words: {len(words)}, offset/annot: {len(offsets)}/{len(annot)}"      
            
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
                
    if count % 12 == 0:
        with open(f"./data/helsearkiv/batch/ner/{BATCH}/{count // 12}.pdf", "wb") as file:
            writer.write(file)
        
        
        writer = pypdf.PdfWriter()
        df = pd.DataFrame({
            'Text': merged_entities,
            'Id': [f"{offset[0]}-{offset[1]}"for offset in merged_offsets],
            'MedicalEntity': merged_annots,
            'DCT': None,
            'TIMEX': None,
            'Context': None
        })
        df.to_csv(f"./data/helsearkiv/batch/ner/{BATCH}/{count // 12}.csv")
        
        merged_entities = []
        merged_offsets = []
        merged_annots = []
        
                

df = pd.DataFrame(info_file)
df.to_csv(f'./data/helsearkiv/batch/ner/{BATCH}')
