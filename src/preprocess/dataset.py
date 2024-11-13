import pandas as pd
from typing import List

COLUMN_NAMES = [
    'id', #0
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
               for i, line in enumerate(f):
                if i in range(7) or line.strip() in ['\n', '']:
                    continue
                
                if line.startswith('#'):
                    continue
                sentence = line.split('\t')
                
                offset = 0
                if len(sentence) != 16:
                    offset = -2

                clip = sentence[10+offset].find('[')
                if clip == -1:
                    clip = len(sentence[10+offset])
                    
                row = {
                    'id': sentence[0] if sentence[0] != '_' else "O", #0
                    'Offset': sentence[1] if sentence[1] != '_' else "O", #1
                    'Text': sentence[2] if sentence[2] != '_' else "O", #1
                    'Modality': sentence[8+offset] if sentence[8+offset] != '_' else "O", #8
                    'Polarity': sentence[9+offset] if sentence[9+offset] != '_' else "O", #9,
                    'Medical Entity': sentence[10+offset][:clip] if sentence[10+offset] != '_' else "O", #10,
                    'Temporal Feature': sentence[11+offset] if sentence[11+offset] != '_' else "O", #11,
                    'Entity Relation': sentence[12+offset] if sentence[12+offset] != '_' else "O", #12,
                    'Temporal Relation': sentence[13+offset] if sentence[13+offset]!= '_' else "O", #13,
                    'fk_id': sentence[14+offset] if sentence[14+offset] != '_' else "O"#14
                }
                document.loc[len(document)] = row 
                
            self.__documents.append(document)
    
    def get(self, task:str):
        match task:
            case 'MER':
                return self.__get_docs_by_cols(['id', 'Text', 'Medical Entity'])
            case 'TFE':
                return self.__get_docs_by_cols(['id', 'Text', 'Temporal Feature'])
            case 'TRE':
                return self.__get_docs_by_cols(['id', 'Text', 'Temporal Relation', 'fk_id'])
            case 'ERE':
                return self.__get_docs_by_cols(['id', 'Text', 'Entity Relation', 'fk_id'])
            case _:
                return None
                
    def __get_docs_by_cols(self, cols: List[str]):
        return [doc[cols] for doc in self.__documents]


if __name__ == "__main__":
    manager = DatasetManager(['./data/annotated/journal.tsv', './data/annotated/journal-2.tsv'])
    
    print(manager.get('MER'))