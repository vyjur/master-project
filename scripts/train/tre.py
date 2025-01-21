import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager

print("##### Start training for TRE... ######")

folder = "./scripts/train/config/tre/"
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
    save_directory = "./models/tre/" + conf.replace(".ini", "")
    ner = TRExtract(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task.")

print("Process finished!")
