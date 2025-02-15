import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TLINK-context... ######")

config = "./scripts/tee/config/context/config.ini"

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


for i, context in enumerate([35, 50, 65, 80]):
    print(f"###### ({i}) Training for context length: {context}")
    save_directory = f"./models/tre/tlink/context/" +  str(context)
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    
    manager = DatasetManager(entity_files, relation_files, window_size=context)

    ner = TRExtract(
        config_file=config,
        manager=manager,
        task=Dataset.TLINK,
        save_directory=save_directory,
    )
    print("Finished with this task. \n \n")

print("Process finished!")
