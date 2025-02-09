import re
import pandas as pd
from typing import List
from structure.enum import Dataset

### INFO: This is old version in annotated folder
COLUMN_NAMES = [
    "id",  # 0
    "sentence_id",
    "Offset",  # 1
    "Text",  # 2
    "Modality",  # 8
    "Polarity",  # 9,
    "Medical Entity",  # 10,
    "Temporal Feature",  # 11,
    "Entity Relation",  # 12,
    "Temporal Relation",  # 13,
    "fk_id",  # 14
]

### INFO: New annotated_MTSamples
COLUMN_NAMES = [
    "id",  # 0,
    "sentence_id",
    "Offset",  # 1
    "Text",  # 2
    "MER",  # 4 ,
    "TRE_DCT",  # 3,
    "TRE_TLINK",  # 5,
    "fk_id",  # 6
]

### INFO: New synthetic annotated cols
COLUMN_NAMES = [
    "id", # 0,
    "sentence_id", 
    "Offset", #1
    "Text", #2
    "TRE_DCT", #3
    "MER", #4, 
    "TRE_TLINK", #5
    "fk_id", #6
]

COLUMN_NAMES_DICT = {
    "id":0, # 0,
    "sentence_id" : 0, 
    "Offset": 1, #1
    "Text": 2, #2
    "TRE_DCT": 3, #3
    "MER": 4, #4, 
    "TRE_TLINK": 5, #5
    "fk_id": 6, #6
}

class DatasetManager:
    def __init__(self, entity_files: List[str], relation_files: List[str]):

        print("### Processing the files:")
        all_entity_df = []
        for file in entity_files:
            df = pd.read_csv(file, delimiter=',')
            if len(all_entity_df) > 0:
                all_entity_df = pd.concat([all_entity_df, df])
            else:
                all_entity_df = df
        self.__entity_df = all_entity_df.reset_index()
        
        # TODO: config?
        tokens_expanded = self.__entity_df["Text"].str.split().explode().tolist()  # Flatten the token list
        # Define the context window size
        window_size = 50

        # Function to get the context window for each row (respecting original dataset)
        def get_context_window(idx):
            # Get the flattened index of the current token
            token_start_idx = sum(len(t.split()) for t in df["token"][:idx])
            
            # Extract the context window from the flattened list
            context_start = max(0, token_start_idx - window_size // 2)
            context_end = token_start_idx + window_size // 2
            return " ".join(tokens_expanded[context_start:context_end])

        # Apply function to each row
        self.__entity_df["context_window"] = df.index.map(get_context_window)

        all_relation_df = []
        for file in relation_files:
            df = pd.read_csv(file, delimiter=',')
            if len(all_relation_df) > 0:
                all_relation_df = pd.concat([all_relation_df, df])
            else:
                all_relation_df = df
        self.__relation_df = all_relation_df
            
    def get(self, task: Dataset):
        match task:
            case Dataset.NER:
                return self.__get_ent_by_cols(["Id", "Text", "MedicalEntity", "TIMEX", "Context"])
            case Dataset.TRE_DCT:
                result = self.__get_ent_by_cols(["Id", "Text", "DCT", "Context"])
                return result[result["DCT"].notna()]
            case Dataset.TRE_TLINK:
                return self.__get_tlink()

    def __get_ent_by_cols(self, cols: List[str]):
        return self.__entity_df[cols]
    
    def __get_tlink(self):
        return self.__relation_df

#class DatasetManager:
    #def __init__(self, files: List[str]):
        #self.__documents = []

        #print("### Processing the files:")
        #for file in files:
            #document = pd.DataFrame(columns=COLUMN_NAMES)  # type: ignore
            #print(file)

            #with open(file, encoding="UTF-8") as f:
                #connect = False
                #prev_id = -1
                #for i, line in enumerate(f):
                    #if i in range(6) or line.strip() in ["\n", ""]:
                        #continue

                    #if line.startswith("#"):
                        #continue
                    #sentence = line.split("\t")                    
                    #if len(sentence) > COLUMN_NAMES_DICT['fk_id']:
                        #fk_id = sentence[COLUMN_NAMES_DICT['fk_id']]
                        #clip = fk_id.find("[")
                        #if clip != -1:
                            #fk_id = fk_id[:clip]

                        #fk_id = fk_id.split("|")
                    #else:
                        #fk_id = [None]
                    #t_relation = sentence[COLUMN_NAMES_DICT['TRE_TLINK']].split("|")

                    #clip = sentence[COLUMN_NAMES_DICT['MER']].find("[")
                    #if clip == -1:
                        #clipper = len(sentence[COLUMN_NAMES_DICT['MER']])
                    #else:
                        #clipper = clip

                    #if not connect and clip != -1:
                        #connect = True
                        #prev_id = sentence[COLUMN_NAMES_DICT['id']]
                    #elif clip == -1:
                        #connect = False

                    #if connect and prev_id != sentence[0]:
                        #for i in range(len(document)):
                            #if document.loc[i]["id"] == prev_id:
                                #document.loc[i, "Text"] = (
                                    #document.loc[i, "Text"] + " " + sentence[COLUMN_NAMES_DICT['Text']]
                                #)
                    #else:
                        #for i, id in enumerate(fk_id):
                            #row = {
                                #"id": sentence[COLUMN_NAMES_DICT['id']] if sentence[COLUMN_NAMES_DICT['id']] != "_" else "O",  # 0
                                #"sentence_id": int(sentence[COLUMN_NAMES_DICT['id']].split("-")[0]),
                                #"Offset": sentence[COLUMN_NAMES_DICT['Offset']]
                                #if sentence[COLUMN_NAMES_DICT['Offset']] != "_"
                                #else "O",  # 1
                                #"Text": sentence[COLUMN_NAMES_DICT['Text']] if sentence[COLUMN_NAMES_DICT['Text']] != "_" else "O",  # 1
                                #"MER": sentence[COLUMN_NAMES_DICT['MER']][:clipper]
                                #if sentence[COLUMN_NAMES_DICT['MER']] != "_"
                                #else "O",  # 4
                                #"TRE_DCT": re.sub(r"\[.*?\]", "", sentence[COLUMN_NAMES_DICT['TRE_DCT']])
                                #if sentence[COLUMN_NAMES_DICT['TRE_DCT']] != "_"
                                #else "O",  # 3,
                                #"TRE_TLINK": re.sub(r"\[.*?\]", "", t_relation[i])
                                #if t_relation[i] != "_"
                                #else None,  # 13,
                                #"fk_id": id if id != "_" else None,  # 14
                            #}
                            #document.loc[len(document)] = row

            #self.__documents.append(document)

    #def get(self, task: Dataset):
        #match task:
            #case Dataset.NER:
                #return self.__get_docs_by_cols(["id", "sentence_id", "Text", "MER"])
            #case Dataset.TRE_DCT:
                #result = self.__get_docs_by_cols(["id", "Text", "TRE_DCT"])
                #return [doc[doc["TRE_DCT"] != "O"] for doc in result]
            #case Dataset.TRE_TLINK:
                #return self.__get_docs_by_cols(["id", "Text", "TRE_TLINK", "fk_id"])
            #case Dataset.SENTENCES:
                #return self.__get_sentences()

    #def __get_docs_by_cols(self, cols: List[str]):
        #return [doc[cols].drop_duplicates() for doc in self.__documents]

    #def __get_sentences(self):
        #return [
            #doc.groupby("sentence_id")["Text"].apply(lambda x: " ".join(x))
            #for doc in self.__documents
        #]


if __name__ == "__main__":
    manager = DatasetManager(
        ["./data/annotated/journal.tsv", "./data/annotated/journal-2.tsv"]
    )

    print(manager.get(Dataset.TRE_DCT))
