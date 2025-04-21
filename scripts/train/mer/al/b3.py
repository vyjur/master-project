import os
from textmining.mer.setup import MERecognition
from preprocess.dataset import DatasetManager

print("##### Start training for NER... ######")

folder = "./scripts/train/ner/config/model/"
configs = os.listdir(folder)

folder_path = "./data/helsearkiv/annotated/entity/"

entity_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

folder_path = "./data/helsearkiv/batch/ner/final/"

batch_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.split("-")[0] in ['b1', 'b2']
]

entity_files.extend(batch_files)

folder_path = "./data/helsearkiv/annotated/relation/"

relation_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

manager = DatasetManager(entity_files, relation_files)

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
    if conf != 'b-bert.ini':
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    if os.path.isdir(folder + conf):
        continue
    save_directory = "./models/ner/model/" + conf.replace(".ini", "")
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    ner = MERecognition(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        test_manager=test_manager
    )
    print("Finished with this task. \n \n")

print("Process finished!")
