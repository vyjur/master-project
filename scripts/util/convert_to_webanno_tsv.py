import os 
import pandas as pd
import textwrap
BATCH = 1

out_folder_path = f"./data/helsearkiv/batch/ner/{BATCH}-webanno/"
in_folder_path = f"./data/helsearkiv/batch/ner/{BATCH}-local/"

os.mkdir(out_folder_path)

entity_files = [
    (in_folder_path + f, f)
    for f in os.listdir(in_folder_path)
    if os.path.isfile(os.path.join(in_folder_path, f))
]

for i, (path, file) in enumerate(entity_files[:10]):
        
    if ".pdf" in file or not "56" in file:
        continue
    
    content = textwrap.dedent("""\
        #FORMAT=WebAnno TSV 3.3
        #T_SP=org.dkpro.core.api.pdf.type.PdfPage|height|pageNumber|width
        #T_SP=webanno.custom.MedicalEntity|DCT|MedicalEntity|TIMEX
        #T_RL=webanno.custom.TLINK|TLINK|BT_webanno.custom.MedicalEntity


    """)
    
    df = pd.read_csv(path)
    offset = 0
    sentence_id = 1
    word_id = 1
    annot_id = 1
    
    end_index = offset
    
    with open(path.replace(in_folder_path, out_folder_path).replace(file, f"b{BATCH}-{file}").replace(".csv", ".tsv"), 'w') as out:
        full_text = f"{" ".join(df['Text'].apply(lambda x: str(x).strip()))}".replace('[UNK]', '').replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '')
        content += "#Text=" +full_text
        content += "\n"
        
        for j, row in df.iterrows():
            row['Text'] = str(row['Text']).replace('[UNK]', '').replace('[SEP]', '').replace('[CLS]', '').replace('[PAD]', '')
            words = row['Text'].split()
            temp = ''
            if len(words) > 1:
                temp = f"[{word_id}]"
                word_id += 1
            for word in words:
                word = word.strip()
                
                start_index = end_index + full_text[end_index:].index(word)
                end_index = start_index + len(word)
                content += f'{sentence_id}-{word_id}\t{start_index}-{end_index + 1}\t{word}\t_\t_\t_\t_\t{row['MedicalEntity'] if row['MedicalEntity'] != "O" else '_'}{temp}\t_\t_\t_\n' 
        
        out.write(content)