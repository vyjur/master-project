import os
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager

print("##### Start training for TRE... ######")
configs = os.listdir("./scripts/train/config/ner")

folder_path = "./data/annotated/"
files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]
manager = DatasetManager(files)


for i, conf in enumerate(configs):
    print(f"###### ({i}) Training for configuration file: {conf}")
    save_directory = conf.replace(".ini", "")
    ner = TRExtract(config_file=conf, manager=manager, save_directory=save_directory)
    print("Finished with this task.")

print("Process finished!")
