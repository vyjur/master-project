import os
from textmining.ner.setup import NERecognition
from preprocess.dataset import DatasetManager

print("##### Start training for NER... ######")
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
    save_directory = "./models/" + conf.replace(".ini", "")
    ner = NERecognition(
        config_file="./scripts/train/config/ner/" + conf,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task.")

print("Process finished!")
