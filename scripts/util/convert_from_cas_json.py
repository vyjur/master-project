from cassis import *
import pandas as pd
import os

print(os.getcwd())
BATCH = 3
folder_path = f"./data/helsearkiv/batch/ner/{BATCH}-json/"
id = 0

folder_path = f"./data/helsearkiv/test_dataset/annotation/"
files = [
    (f.replace('.pdf', ''), folder_path + f + "/admin.json")
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f, "admin.json"))
]

context_window = 50  # Number of characters before and after each token

for name, file in files:
    print(file)
    empty = 0
    with open(file, 'rb') as f:
        cas = load_cas_from_json(f)
        
    raw_data = []
    for token in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"):
        curr_token = {
           'Text': token.get_covered_text(),
            'Id': f"{id}-{token.begin}-{token.end}",
            'MedicalEntity': None,
            'DCT': None,
            'TIMEX': None,
            'Context': None
        }

        if len(token.get_covered_text()) == 1:
            empty += 1
        curr_entity = None
        try:
            medical_entities = cas.select("webanno.custom.MedicalEntity")  # Get all MedicalEntity annotations
        except:
            continue
        for entity in medical_entities:
            if entity.begin <= token.begin < entity.end:  # Token starts within the entity span
                empty = 0
                curr_token['Text'] = entity.get_covered_text()
                curr_token['Id'] = f"{id}-{entity.begin}-{entity.end}"
                curr_token['MedicalEntity'] = entity.MedicalEntity
                curr_token['DCT'] = entity.DCT
                curr_token['TIMEX'] = entity.TIMEX  
                curr_entity = entity
                break  # No need to continue checking once we find the first match

        if curr_entity is None:
            curr_entity = token
        for search_token in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"):
            if curr_entity.begin <= search_token.begin and search_token.end <= curr_entity.end:
                # Extract token text
                token_text = search_token.get_covered_text()
        
                # Get context by slicing around the token
                start = max(0, search_token.begin - context_window)  # Prevent negative index
                end = min(len(cas.sofa_string), search_token.end + context_window)  # Prevent out-of-bounds
                curr_token['Context'] = cas.sofa_string[start:end]
        raw_data.append(curr_token)

        if curr_token['MedicalEntity'] is None and curr_token['DCT'] is None and curr_token['TIMEX'] is None:
            empty += 1

        if empty > 5000:
            print("BREAK")
            break
    
    print(file, len(raw_data))

    if len(raw_data) > 0:
        df = pd.DataFrame(raw_data).drop_duplicates()
        df.to_csv(f'./data/helsearkiv/test_dataset/csv/entity/{name}.csv')
    
    raw_data = []
        
    # Iterate over each MedicalEntity annotation
    try:
        cas.select("webanno.custom.TLINK")
    except:
        continue
    for link in cas.select("webanno.custom.TLINK"):
        curr_link = {
            'FROM': link.Dependent.get_covered_text(),
            'FROM_Id': f"{id}-{link.Dependent.begin}-{link.Dependent.end}",
            'FROM_CONTEXT': None,
            'TO': link.Governor.get_covered_text(),
            'TO_Id': f"{id}-{link.Governor.begin}-{link.Governor.end}",
            'TO_CONTEXT': None,
            'RELATION': link.TLINK
        }
        
        # Find tokens inside this MedicalEntity annotation
        contexts = []
        for entity in [link.Dependent, link.Governor]:
            for token in cas.select("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"):
                if entity.begin <= token.begin and token.end <= entity.end:
                    # Extract token text
                    token_text = token.get_covered_text()
        
                    # Get context by slicing around the token
                    start = max(0, token.begin - context_window)  # Prevent negative index
                    end = min(len(cas.sofa_string), token.end + context_window)  # Prevent out-of-bounds
                    context = cas.sofa_string[start:end]
                    contexts.append(context)
                    #print(f"  Token: {token_text}, Begin: {token.begin}, End: {token.end}, Context: {context}")
        
        curr_link['FROM_CONTEXT'] = contexts[0]
        curr_link['TO_CONTEXT'] = contexts[1]
        raw_data.append(curr_link)

    df = pd.DataFrame(raw_data).drop_duplicates()
    df.to_csv(f'./data/helsearkiv/test_dataset/csv/relation/{name}.csv')
    id += 1
