import os
from textmining.tee.setup import TEExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

BATCH=4

print("##### Start training for TEE... ######")

# Info: Change here

folder = f"./scripts/train/tee/config/input/"
configs = os.listdir(folder)

#folder_path = "./data/helsearkiv/annotated/entity/"

folder_path = "./data/helsearkiv/batch/tee/base/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/batch/tee/"

batch_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and "final" in f
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
    if not "xml" in conf:
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tee/model/" + conf.replace(".ini", "") + f"/b{BATCH}/"
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    ner = TEExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        test_manager=test_manager
    )
    print("Finished with this task. \n \n")

print("Process finished!")
