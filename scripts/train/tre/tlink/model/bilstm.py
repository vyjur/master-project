import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "tlink"

folder = f"./scripts/train/tre/{TRE_TYPE}/config/model/"
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


# folder_path = "./data/helsearkiv/batch/tlink/"

# batch_files = [
#     folder_path + f
#     for f in os.listdir(folder_path)
#     if os.path.isfile(os.path.join(folder_path, f))
# ]

# relation_files.extend(batch_files)

manager = DatasetManager(entity_files, relation_files, window_size=512)

folder_path = "./data/helsearkiv/test_dataset/csv/relation/"

relation_files = [
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
    if os.path.isdir(folder + conf) or conf != 'a-bilstm.ini':
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tre/{TRE_TYPE}/model/" + conf.replace(".ini", "")
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        task=Dataset.TLINK,
        test_manager=test_manager
    )
    print("Finished with this task. \n \n")

print("Process finished!")
