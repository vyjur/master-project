import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "dtr"

folder = f"./scripts/train/tre/{TRE_TYPE}/config/model/"
configs = os.listdir(folder)

folder_path = "./data/helsearkiv/annotated/entity-w-dct/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

## MEDICAL ENTITY batch
#folder_path = "./data/helsearkiv/batch/ner/final/"

#batch_files = [
    #folder_path + f
    #for f in os.listdir(folder_path)
    #if os.path.isfile(os.path.join(folder_path, f)) and "b4" not in f
#]

#entity_files.extend(batch_files)

# DTR batch
folder_path = "./data/helsearkiv/batch/dtr-w-dct/"

batch_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and "final" in f and f.split("-")[0] in ["1", "2", "3"]
]

entity_files.extend(batch_files)

folder_path = "./data/helsearkiv/annotated/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

manager = DatasetManager(entity_files, relation_files, window_size=50)

### TEST DATA
folder_path = "./data/helsearkiv/test_dataset/csv/entity/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/test_dataset/csv/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

test_manager = DatasetManager(entity_files, relation_files)

for i, conf in enumerate(configs):
    if os.path.isdir(folder + conf) or conf != 'b-bert.ini':
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tre/{TRE_TYPE}/model/" + conf.replace(".ini", "")
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        task=Dataset.DTR,
        test_manager=test_manager
    )
    print("Finished with this task. \n \n")

print("Process finished!")
