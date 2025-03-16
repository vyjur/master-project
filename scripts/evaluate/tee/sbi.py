import os
import pandas as pd
from structure.enum import Dataset
from preprocess.dataset import DatasetManager
from sklearn.metrics import classification_report
from textmining.tee.setup import TEExtract
from sklearn.metrics import classification_report

import pypdf
files = ['0c6bfbe4-f987-4022-81d3-f8183ce5f3e6_22d698ed-2284-40a0-aa0d-11338341fdd1.pdf']

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

manager = DatasetManager(entity_files, relation_files, False, False)

config_file = "./scripts/evaluate/tee/config.ini"
save_directory = "./models/tee/model/b-bert/"
tee = TEExtract(config_file=config_file, manager=manager, save_directory=save_directory, rules=False)
tee.set_dct('2025-02-10')

cities = pd.read_csv('./scripts/evaluate/tee/no-cities.csv')['city'].values

def contains_any(lst, s):
    
    for item in lst:
        if item.lower() in s:
            print("item:", item, "context:", s)
    return any(item.lower() in s for item in lst)

def sectime(row):
    lower_context = row['context'].lower()

    if contains_any(cities, lower_context):
        print("CITY")
        return "SECTIME"
    
    left = ["Status presens", "Innl. dato", "Inn:", "Dato:", "Diktert dato", "Dikt", "kl. us."]
    left = ["Status presens", "Innl. dato", "Diktert dato", "Dikt", "kl. us."]

    right = ["ved dr.", "v/ dr.", "HENVISNING TIL", "tilsett fra", "DAGNOTAT", "POLIKLINISK NOTAT", "Søknadsskjema", "INNKOMSTJOURNAL", "TILSYNSNOTAT", "OPERASJONSBESKRIVELSE", "Korrigering av opr.beskrivelse", "JOURNALNOTAT", "SLUTTNOTAT"]
    lower_left = [ i.lower() for i in left]
    lower_right = [ i.lower() for i in right]
    
    start_idx = lower_context.index(row['text'].lower())
    before_context = lower_context[:start_idx]
    after_context = lower_context[start_idx:start_idx + len(row['text'])]
    
    if contains_any(lower_left, before_context):
        print("BEFORE")
        return "SECTIME"
    elif contains_any(lower_right, after_context):
        print("AFTER")
        return "SECTIME"
    return "DATE"

for file in files:
    
    reader = pypdf.PdfReader("./data/helsearkiv/journal/" + file)
    for i, page in enumerate(reader.pages[:37]):
        output = tee.run(page.extract_text())
        for j, row in output.iterrows():
            if row['type'] == 'DATE' and not row['text'].isalpha():
                text = row['text']
                context = row['context']
                
                start = context.find(text)
                end = start + len(text)

                # Extract ±20 character window
                window_start = max(0, start - 20)
                window_end = min(len(context), end + 20)

                # Get the surrounding context
                surrounding_text = context[window_start:window_end]

                # Highlight the text
                highlighted_context = surrounding_text.replace(text, f"<TAG>{text}</TAG>")
                
                row['context'] = surrounding_text
                
                result = sectime(row)
                pred = tee.predict_sectime(row)
                print(i, "-", j, "###########################################################################")
                print(f"\nText: {row['text']}, TIMEX: {row['type']}, SECTIME: {result}, MODEL: {pred}")
                
                print(f"Context: {highlighted_context}")
                
                
print("####!!!!!ALL_DATASET!!!!!####")

dataset = manager.get(Dataset.TEE)

pred = []
target = []
outputs = []

for _, row in dataset.iterrows():
    if row['Text'].replace(" ", "").isalpha() or 'ICD' in row['Context']:
        continue
    
    if len(result) <= 0:
        continue
    
    text = row['Text']
    context = row['Context']
    
    start = context.find(text)
    end = start + len(text)

    # Extract ±20 character window
    window_start = max(0, start - 20)
    window_end = min(len(context), end + 20)

    # Get the surrounding context
    surrounding_text = context[window_start:window_end]

    # Highlight the text
    highlighted_context = surrounding_text.replace(text, f"<TAG>{text}</TAG>")
        
    e = {
        "text": row['Text'],
        "context": surrounding_text
    }
    result = sectime(e)
    output = tee.predict_sectime(e)
    outputs.append(output[0][0].replace("DCT", "SECTIME"))
    pred.append(result)
    
    print(i, "-", j, "###########################################################################")
    print(f"\nText: {e['text']}, TIMEX: {row['TIMEX']}, SECTIME: {result}, MODEL: {output}")
    
    
    
    print(f"Context: {highlighted_context}")
    target.append(row['TIMEX'].replace('DCT', 'SECTIME'))
    

print(classification_report(target, pred))

print(classification_report(target, outputs))
                
                
                
                
                
                
                
                

    