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

tee = TEExtract(rules=False)
tee.set_dct('2025-02-10')
    
manager = DatasetManager(entity_files, relation_files, False)

dataset = manager.get(Dataset.TEE)

print("### Without more rules")

target = []
pred = []
for i, data in dataset.iterrows():
    output = tee.run([data['Text']])[0]
    if not output.empty:
        pred.append(output["type"].values[0])
        target.append(data['TIMEX'].replace('DCT', 'DATE'))
    else:
        pred.append(None)
        target.append(data['TIMEX'].replace('DCT', 'DATE'))

print(classification_report(target, pred))

print("### With handcrafted rules")
tee = TEExtract(rules=True)
tee.set_dct('2025-02-10')

target = []
pred = []
for i, data in dataset.iterrows():
    output = tee.run([data['Text']])[0]
    if not output.empty:
        pred.append(output["type"].values[0])
        target.append(data['TIMEX'].replace('DCT', 'DATE'))
    else:
        pred.append(None)
        target.append(data['TIMEX'].replace('DCT', 'DATE'))

print(classification_report(target, pred))