import os
from textmining.ner.setup import NERecognition
from textmining.tee.setup import TEExtract
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset


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

manager = DatasetManager(entity_files[:5], relation_files[:5])

print("##### Check NER... ######")

file = "./scripts/train/config/ner/a-bilstmcrf.ini"

try:
    ner = NERecognition(
        config_file=file,
        manager=manager,
    )
except:
    pass

print("##### Check TEE... ######")

file = "./scripts/train/config/tee/a-bilstm.ini"
try:
    ner = TEExtract(
        config_file=file,
        manager=manager,
    )
except:
    pass

print("##### Check DTR... ######")

file = "./scripts/train/config/tre/dtr/a-bilstm.ini"
try:
    ner = TRExtract(
        config_file=file,
        manager=manager,
        task=Dataset.DTR
    )
except Exception as error:
    print(error)
    
print("##### Check TLINK... ######")

file = "./scripts/train/config/tre/tlink/a-bilstm.ini"

try:
    ner = TRExtract(
        config_file=file,
        manager=manager,
        task=Dataset.TLINK
    )
except Exception as error:
    print(error)