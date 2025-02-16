import os
from preprocess.dataset import DatasetManager
from transformers import AutoTokenizer
from structure.enum import Dataset

SIZE = 50

print(f"WINDOW SIZE: {SIZE}")

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

tokenizer = AutoTokenizer.from_pretrained("ltg/norbert3-small")

manager = DatasetManager(entity_files, relation_files, window_size=SIZE)

print("#### DTR / TEE Context")

dataset_ner = manager.get(Dataset.NER)
dataset_ner = dataset_ner[dataset_ner['MedicalEntity'].notna() | dataset_ner['TIMEX'].notna()].reset_index()

sentences = [ len(tokenizer(row["Context"].replace(row["Text"], f"<TAG>{row['Text']}</TAG>"))['input_ids']) for _, row in dataset_ner.iterrows() ]

print(f"Mean: {sum(sentences)/len(sentences)}, Max: {max(sentences)}, Min: {min(sentences)}")
print("\n")
print("#### TLINK Context")
dataset_tre = manager.get(Dataset.TLINK)

sentences = []

for i, rel in dataset_tre.iterrows():
                    
    e_i = dataset_ner[dataset_ner['Id'] == rel['FROM_Id']]
    e_j = dataset_ner[dataset_ner['Id'] == rel['TO_Id']]
    
    if len(e_i) > 0 and len(e_j) > 0:
        e_i = e_i.iloc[0]
        e_j = e_j.iloc[0]
    else:
        continue

    sentence_i = e_i['Context'].replace(e_i['Text'], f"<TAG>{e_i['Text']}</TAG>")
    sentence_j = e_j['Context'].replace(e_j['Text'], f"<TAG>{e_j['Text']}</TAG>")
    
    words = f"{sentence_i} [SEP] {sentence_j}"
    
    sentences.append(len(tokenizer(words)['input_ids']))

print(f"Mean: {sum(sentences)/len(sentences)}, Max: {max(sentences)}, Min: {min(sentences)}")