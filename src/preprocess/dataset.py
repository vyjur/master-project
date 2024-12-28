import pandas as pd
from typing import List
from structure.enum import Dataset

COLUMN_NAMES = [
    'id', #0
    'sentence_id',
    'Offset', #1
    'Text', #2
    'Modality', #8
    'Polarity', #9,
    'Medical Entity', #10,
    'Temporal Feature', #11,
    'Entity Relation', #12,
    'Temporal Relation', #13,
    'fk_id', #14
]

class DatasetManager:
    def __init__(self, files:List[str]):
        self.__documents = []
        for file in files:
            document = pd.DataFrame(columns=COLUMN_NAMES)
            
            with open(file, encoding='UTF-8') as f:
                connect = False
                prev_id = -1
                for i, line in enumerate(f):
                    if i in range(7) or line.strip() in ['\n', '']:
                        continue
                    
                    if line.startswith('#'):
                        continue
                    sentence = line.split('\t')
                    
                    offset = 0
                    if len(sentence) != 16:
                        offset = -2
                        
                    fk_id = sentence[14+offset]
                    clip = fk_id.find('[')
                    if clip != -1:
                        fk_id = fk_id[:clip] 

                    fk_id = fk_id.split('|')
                    t_relation = sentence[13+offset].split('|')
                    e_relation = sentence[12+offset].split('|')
                    
                    clip = sentence[10+offset].find('[')
                    if clip == -1:
                        clipper = len(sentence[10+offset])  
                    else:
                        clipper = clip

                    if not connect and clip != -1:
                        connect = True
                        prev_id = sentence[0]
                    elif clip == -1:
                        connect = False
                        
                    if connect and prev_id != sentence[0]:
                        for i in range(len(document)):
                            if document.loc[i]['id'] == prev_id:
                                document.loc[i, 'Text'] = document.loc[i, 'Text'] + ' ' + sentence[2]
                    else:
                        for i, id in enumerate(fk_id):
                            row = {
                                'id': sentence[0] if sentence[0] != '_' else "O", #0
                                'sentence_id': int(sentence[0].split('-')[0]),
                                'Offset': sentence[1] if sentence[1] != '_' else "O", #1
                                'Text': sentence[2] if sentence[2] != '_' else "O", #1
                                'Modality': sentence[8+offset] if sentence[8+offset] != '_' else "O", #8
                                'Polarity': sentence[9+offset] if sentence[9+offset] != '_' else "O", #9,
                                'Medical Entity': sentence[10+offset][:clipper] if sentence[10+offset] != '_' else "O", #10,
                                'Temporal Feature': sentence[11+offset] if sentence[11+offset] != '_' else "O", #11,
                                'Entity Relation': e_relation[i] if e_relation[i] != '_' else "O", #12,
                                'Temporal Relation': t_relation[i] if t_relation[i] != '_' else None, #13,
                                'fk_id': id if id != '_' else None #14
                            }
                            document.loc[len(document)] = row 
                
            self.__documents.append(document)
    
    def get(self, task:Dataset):
        match task:
            case Dataset.MER:
                return self.__get_docs_by_cols(['id', 'sentence_id', 'Text', 'Medical Entity'])
            case Dataset.TFE:
                return self.__get_docs_by_cols(['id', 'Text', 'Temporal Feature'])
            case Dataset.TRE:
                return self.__get_docs_by_cols(['id', 'Text', 'Temporal Relation', 'fk_id'])
            case Dataset.ERE:
                return self.__get_docs_by_cols(['id', 'Text', 'Entity Relation', 'fk_id'])
            case Dataset.SENTENCES:
                return self.__get_sentences()
            case _:
                return None
                
    def __get_docs_by_cols(self, cols: List[str]):
        return [doc[cols].drop_duplicates() for doc in self.__documents]
    
    def __get_sentences(self):
        return [doc.groupby('sentence_id')['Text'].apply(lambda x: ' '.join(x)) for doc in self.__documents]


if __name__ == "__main__":
    manager = DatasetManager(['./data/annotated/journal.tsv', './data/annotated/journal-2.tsv'])
    
    print(manager.get(Dataset.MER))