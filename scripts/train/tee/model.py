import os
from textmining.tee.setup import TEExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TEE... ######")

# Info: Change here

folder = f"./scripts/train/tee/config/model/"
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
    save_directory = f"./models/tee/model/" + conf.replace(".ini", "")
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    ner = TEExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task. \n \n")

print("Process finished!")
