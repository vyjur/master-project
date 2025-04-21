import os
import pypdf
from pipeline.setup import Pipeline

pipeline = Pipeline("./src/pipeline/config.ini")


save_path = './scripts/pipeline/'

FILE_PATH = ""
reader = pypdf.PdfReader(FILE_PATH)

documents = []
print("PAGES:", len(reader.pages))
for j, page in enumerate(reader.pages[3:]):
    documents.append(page.extract_text())    
    pipeline.run(documents, save_path=(save_path), dct_cp=True)
        
print("### FINISHED ###")
