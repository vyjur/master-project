import os
import pypdf
import pandas as pd
import random

noisy = 0
not_noisy = 0

folder_path = './data/helsearkiv/journal/'

files = [
    folder_path + f
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

documents = []
save_path = './data/helsearkiv/evaluate/tem1-mer/'
for i, doc in enumerate(files):
        reader = pypdf.PdfReader(doc)
        for j, page in enumerate(reader.pages):
            documents.append(page.extract_text())
       
num_samples = max(1, int(len(documents) * 0.1))  # Ensure at least 1 document is selected
sampled_documents = random.sample(documents, num_samples)

for i, page in enumerate(sampled_documents):
    
    print(f"#########################{i}/{len(documents)*0.05}#########################")
    print(page)
    
    print(f"Noisy: {noisy} Not noisy: {not_noisy} Percentage: {noisy/(len(sampled_documents)*0.05)}")

    result = int(input("1: noisy, 2: not noisy: "))
    
    if result == 1:
        noisy += 1
    else:
        not_noisy += 1

print(f"Noisy: {noisy} Not noisy: {not_noisy} Percentage: {noisy/(len(sampled_documents)*0.05)}")
 
print("### FINISHED ###")
