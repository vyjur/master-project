import re
import numpy as np
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

SAVE_FOLDER = "./data/helsearkiv/processed"

class DatasetManager:
    # TODO: context-size config
    def __init__(self, entity_files: List[str], relation_files: List[str], load:bool=False, context:bool=True, window_size:int=50):
        if not load:
            print("### Processing the files:")
            all_entity_df = []
            for file in entity_files:
                df = pd.read_csv(file, delimiter=',')
                all_entity_df.append(df)
            self.__entity_df = pd.concat(all_entity_df).reset_index()
            
            tokens_expanded = self.__entity_df["Text"].str.split().explode().tolist()  # Flatten the token list
            # Define the context window size
            
            # Function to get the context window for each row (respecting original dataset)
            # Token splits on words (word-level token window)
            def get_context_window(idx):
                # Get the flattened index of the current token
                token_start_idx = sum(len(str(t).split()) for t in self.__entity_df["Text"][:idx])
                
                # Extract the context window from the flattened list
                context_start = max(0, token_start_idx - window_size)
                context_end = min(len(tokens_expanded), token_start_idx + window_size)
                return " ".join(map(str, tokens_expanded[context_start:context_end]))

            # Apply function to each row
            
            if context:
                # Initialize "Context" column with empty strings or NaNs
                self.__entity_df["Context"] = ''

                # Create a mask for rows where MedicalEntity or TIMEX is not NA
                mask = self.__entity_df["MedicalEntity"].notna() | self.__entity_df["TIMEX"].notna()

                # Apply get_context_window only to the filtered rows
                self.__entity_df.loc[mask, "Context"] = self.__entity_df.loc[mask].index.to_series().apply(get_context_window)

            all_relation_df = []
            for file in relation_files:
                df = pd.read_csv(file, delimiter=',')
                all_relation_df.append(df)
            self.__relation_df = pd.concat(all_relation_df)

            self.__entity_df.to_csv(f"{SAVE_FOLDER}/entity_{window_size}.csv")
            self.__relation_df.to_csv(f"{SAVE_FOLDER}/relation.csv")
        else:
            print("Loading dataset")
            self.__entity_df = pd.read_csv(f"{SAVE_FOLDER}/entity_{window_size}.csv")
            self.__relation_df = pd.read_csv(f"{SAVE_FOLDER}/relation.csv")
            
    def get(self, task: Dataset):
        match task:
            case Dataset.NER:
                return self.__get_ent_by_cols(["Id", "Text", "MedicalEntity", "TIMEX", "Context"])
            case Dataset.DTR:
                result = self.__get_ent_by_cols(["Id", "Text", "DCT", "Context"])
                return result[result["DCT"].notna()]
            case Dataset.TEE:
                result = self.__get_ent_by_cols(["Id", "Text", "TIMEX", "Context"])
                return result[result["TIMEX"].notna()]
            case Dataset.TLINK:
                return self.__get_tlink()

    def __get_ent_by_cols(self, cols: List[str]):
        return self.__entity_df[cols]
    
    def __get_tlink(self):
        return self.__relation_df

if __name__ == "__main__":
    manager = DatasetManager(
        ["./data/annotated/journal.tsv", "./data/annotated/journal-2.tsv"]
    )

    print(manager.get(Dataset.DTR))
