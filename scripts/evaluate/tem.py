import os
import pypdf
import pandas as pd
from pipeline.setup import Pipeline

pipeline = Pipeline("./src/pipeline/config.ini")

folder_path = './data/helsearkiv/journal/'

patients_df = pd.read_csv('./data/helsearkiv/patients.csv')

files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

save_path = './data/helsearkiv/evaluate/tem1/'
for i, doc in enumerate(files):
    if doc.split("_")[1].strip().replace('.pdf', "") in patients_df['journalidentifikator'].to_list():
        print(f"- Executing the pipeline for {doc}")
        reader = pypdf.PdfReader(doc)
        documents = []
        print("PAGES:", len(reader.pages))
        for j, page in enumerate(reader.pages):
            documents.append(page.extract_text())
        
        doc = doc.replace(".pdf", "/")
        os.makedirs(save_path + doc, exist_ok=True)
        pipeline.run(documents, save_path=(save_path + doc))
        break
        
print("### FINISHED ###")