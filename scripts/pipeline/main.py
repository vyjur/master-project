import os
import pypdf
import pandas as pd
from pipeline.setup import Pipeline

# TODO: add your file name and the path you want to save here
doc = ""
save_path = ""

pipeline = Pipeline("./src/pipeline/config.ini")

print(f"- Executing the pipeline for {doc}")
reader = pypdf.PdfReader(doc)
documents = []
print("PAGES:", len(reader.pages))
for j, page in enumerate(reader.pages):
    documents.append(page.extract_text())

doc = doc.replace(".pdf", "/")
os.makedirs(save_path + doc, exist_ok=True)
pipeline.run(documents, save_path=(save_path + doc), step="MER")
    
print("### FINISHED ###")
