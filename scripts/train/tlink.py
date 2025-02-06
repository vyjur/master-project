import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from structure.enum import Dataset

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "tlink"

folder = f"./scripts/train/config/tre/{TRE_TYPE}/"
configs = os.listdir(folder)
folder_path = "./data/synthetic/annotated/annotation/"

files = [
    folder_path + f + "/admin.tsv"
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f, "admin.tsv"))
]
manager = DatasetManager(files)

for i, conf in enumerate(configs):
    if os.path.isdir(folder + conf):
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tre/{TRE_TYPE}/" + conf.replace(".ini", "")
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
        task=Dataset.TRE_TLINK
    )
    print("Finished with this task.")

print("Process finished!")
