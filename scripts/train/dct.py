import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "dct"

folder = f"./scripts/train/config/tre/{TRE_TYPE}/"
configs = os.listdir(folder)

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

for i, conf in enumerate(configs):
    if os.path.isdir(folder + conf):
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tre/{TRE_TYPE}/" + conf.replace(".ini", "")
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        task=Dataset.TRE_DCT
    )
    print("Finished with this task.")

print("Process finished!")
