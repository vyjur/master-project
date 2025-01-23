import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager

print("##### Start training for TRE... ######")

# Info: Change here
TRE_TYPE = "tlink"

folder = f"./scripts/train/config/tre/{TRE_TYPE}"
configs = os.listdir(folder)
folder_path = "./data/annotated/"
files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
manager = DatasetManager(files)

for i, conf in enumerate(configs):
    if conf != 'c-bert-bilstm.ini':
        continue
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = f"./models/tre/{TRE_TYPE}/" + conf.replace(".ini", "")
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task.")

print("Process finished!")
