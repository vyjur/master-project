import os
import pypdf
import pandas as pd

folder_path = "./data/helsearkiv/journal/"

safe_patients_id = [
    {
        "id": 0,
        "file": "0F978AAC-681F-4A61-9F82-0D00316D8FDC",
        "start": 5,
        "stop": 36
    },
   {
        "id": 3,
        "file": "27C4A575-CAC3-4C87-A353-D96DC09A6ACD",
        "start": 8,
        "stop": 32
    },{
        "id": 4,
        "file": "2F46B2F7-63B7-47D8-8B8C-31962164D939",
        "start": 5,
        "stop": 49
    },
]

journal_files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

output_path = './data/helsearkiv/safe-patients/'

for file in journal_files:
    for patient in safe_patients_id:
        if patient["file"] in file:
            reader = pypdf.PdfReader(file)
            writer = pypdf.PdfWriter()
            
            for page in reader.pages[patient['start']-1:patient['stop']]:
                writer.add_page(page)
                
            with open(output_path + str(patient['id']) + ".pdf", "wb") as output_pdf:
                writer.write(output_pdf)