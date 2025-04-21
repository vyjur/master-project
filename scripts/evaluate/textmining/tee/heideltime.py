import os
from structure.enum import Dataset
from preprocess.dataset import DatasetManager
from sklearn.metrics import classification_report
from textmining.tee.setup import TEExtract

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

manager = DatasetManager(entity_files, relation_files, False, False)

config_file = "./scripts/evaluate/tee/config.ini"
save_directory = "./models/tee/model/b-bert"
tee = TEExtract(config_file=config_file, manager=manager, save_directory=save_directory, rules=False)
tee.set_dct('2025-02-10')
    

dataset = manager.get(Dataset.TEE)

print("### Without more rules")

target = []
pred = []
for i, data in dataset.iterrows():
    output = tee.run(data['Text'])
    targ = data['TIMEX'].replace('DCT', 'DATE')
    if targ not in ['DATE', 'DURATION']:
        continue
    try:
        if not output.empty:
            pred.append(output["type"].values[0])
            target.append(targ)
        else:
            pred.append("O")
            target.append(targ)
    except:
        pass

print(classification_report(target, pred))

print("### With handcrafted rules")
tee = TEExtract(config_file=config_file, manager=manager, save_directory=save_directory, rules=True)
tee.set_dct('2025-02-10')

target = []
pred = []
for i, data in dataset.iterrows():
    output = tee.run(data['Text'])
    targ = data['TIMEX'].replace('DCT', 'DATE')
    if targ not in ['DATE', 'DURATION']:
        continue
    try:
        if not output.empty:
            pred.append(output["type"].values[0])
            target.append(targ)
        else:
            pred.append("O")
            target.append(targ)
    except:
        pass
    

print(classification_report(target, pred))