import os
from textmining.tee.setup import TEExtract
from preprocess.dataset import DatasetManager

print("##### Start training for TEE-context... ######")

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
    save_directory = f"./models/tee/context/" +  str(context)
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    
    manager = DatasetManager(entity_files, relation_files, window_size=context)

    ner = TEExtract(
        config_file=config,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task. \n \n")

print("Process finished!")
