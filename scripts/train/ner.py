import os
from textmining.ner.setup import NERecognition
from preprocess.dataset import DatasetManager

print("##### Start training for NER... ######")

folder = "./scripts/train/config/ner/"
configs = os.listdir(folder)

folder_path = "./data/synthetic/annotated/annotation/"

files = [
    folder_path + f + "/admin.tsv"
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f, "admin.tsv"))
]

manager = DatasetManager(files)

for i, conf in enumerate(configs):
    print(f"###### ({i}) Training for configuration file: {conf}")
    if os.path.isdir(folder + conf):
        continue
    save_directory = "./models/ner/" + conf.replace(".ini", "")
    ner = NERecognition(
        config_file=folder + conf,
        manager=manager,
        save_directory=save_directory,
    )
    print("Finished with this task.")

print("Process finished!")
