import numpy as np
import pandas as pd
from typing import List
from structure.enum import Dataset

COLUMN_NAMES_DICT = {
    "id": 0,  # 0,
    "sentence_id": 0,
    "Offset": 1,  # 1
    "Text": 2,  # 2
    "TRE_DCT": 3,  # 3
    "MER": 4,  # 4,
    "TRE_TLINK": 5,  # 5
    "fk_id": 6,  # 6
}

SAVE_FOLDER = "./data/helsearkiv/processed"


class DatasetManager:
    def __init__(
        self,
        entity_files: List[str],
        relation_files: List[str],
        load: bool = False,
        context: bool = True,
        window_size: int = 50,
    ):
        if not load:
            print("### Processing the files:")
            all_entity_df = []
            for file in entity_files:
                df = pd.read_csv(file, delimiter=",")
                if "page" not in df.columns:
                    df["page"] = np.nan
                if "file" not in df.columns:
                    df["file"] = ""

                if "SECTIME" in df.columns:
                    df = df[
                        [
                            "Text",
                            "Id",
                            "MedicalEntity",
                            "DCT",
                            "TIMEX",
                            "Context",
                            "file",
                            "page",
                            "SECTIME",
                            "SECTIME_context",
                        ]
                    ]
                else:
                    df = df[
                        [
                            "Text",
                            "Id",
                            "MedicalEntity",
                            "DCT",
                            "TIMEX",
                            "Context",
                            "file",
                            "page",
                        ]
                    ]

                all_entity_df.append(df)
            if all_entity_df:  # only concatenate if the list is not empty
                self.__entity_df = pd.concat(all_entity_df).reset_index(drop=True)
            else:
                self.__entity_df = pd.DataFrame(
                    columns=[
                        "Text",
                        "Id",
                        "MedicalEntity",
                        "DCT",
                        "TIMEX",
                        "Context",
                        "file",
                        "page",
                    ]
                )

            tokens_expanded = (
                self.__entity_df["Text"].str.split().explode().tolist()
            )  # Flatten the token list
            # Define the context window size
            self.window_size = window_size

            # Function to get the context window for each row (respecting original dataset)
            # Token splits on words (word-level token window)

            def local_get_context_window(idx):
                # Get the flattened index of the current token
                return self.get_context_window(
                    idx, self.__entity_df["Text"], tokens_expanded
                )

            # Apply function to each row

            if context:
                # Create a mask for rows where MedicalEntity or TIMEX is not NA
                mask = (
                    self.__entity_df["MedicalEntity"].notna()
                    | self.__entity_df["TIMEX"].notna()
                )

                # Create a mask to check for rows where Context is empty
                empty_context_mask = self.__entity_df["Context"] == ""

                # Combine both masks: apply only to rows where Context is empty and MedicalEntity or TIMEX is not NA
                combined_mask = mask & empty_context_mask

                # Apply get_context_window only to the filtered rows
                self.__entity_df.loc[combined_mask, "Context"] = (
                    self.__entity_df.loc[combined_mask]
                    .index.to_series()
                    .apply(local_get_context_window)
                )

            all_relation_df = []
            for file in relation_files:
                df = pd.read_csv(file, delimiter=",")
                df = df[
                    [
                        "FROM",
                        "FROM_Id",
                        "FROM_CONTEXT",
                        "TO",
                        "TO_Id",
                        "TO_CONTEXT",
                        "RELATION",
                    ]
                ]
                all_relation_df.append(df)
            
            if all_relation_df:  # only concatenate if the list is not empty
                self.__relation_df = self.__relation_df = pd.concat(all_relation_df)
            else:
                self.__relation_df = pd.DataFrame(
                    columns=[
                        "FROM",
                        "FROM_Id",
                        "FROM_CONTEXT",
                        "TO",
                        "TO_Id",
                        "TO_CONTEXT",
                        "RELATION",
                    ]
                )


            # self.__entity_df.to_csv(f"{SAVE_FOLDER}/entity_{window_size}.csv")
            # self.__relation_df.to_csv(f"{SAVE_FOLDER}/relation.csv")
        else:
            # print("Loading dataset")
            # self.__entity_df = pd.read_csv(f"{SAVE_FOLDER}/entity_{window_size}.csv")
            # self.__relation_df = pd.read_csv(f"{SAVE_FOLDER}/relation.csv")
            pass

    def get(self, task: Dataset):
        match task:
            case Dataset.NER:
                return self.__get_ent_by_cols(
                    ["Id", "Text", "MedicalEntity", "TIMEX", "DCT", "Context"]
                )
            case Dataset.DTR:
                result = self.__get_ent_by_cols(
                    [
                        "Id",
                        "Text",
                        "MedicalEntity",
                        "TIMEX",
                        "DCT",
                        "Context",
                        "SECTIME",
                        "SECTIME_context",
                    ]
                )
                if "SECTIME" in result.columns:
                    return result[
                        result["DCT"].notna()
                        & result["MedicalEntity"].notna()
                        & result["TIMEX"].isna()
                    ][["Id", "Text", "DCT", "Context", "SECTIME", "SECTIME_context"]]
                return result[
                    result["DCT"].notna()
                    & result["MedicalEntity"].notna()
                    & result["TIMEX"].isna()
                ][["Id", "Text", "DCT", "Context"]]
            case Dataset.TEE:
                result = self.__get_ent_by_cols(["Id", "Text", "TIMEX", "Context"])
                return result[
                    result["TIMEX"].notna() & result["TIMEX"].isin(["DATE", "DCT"])
                ]
            case Dataset.TLINK:
                return self.__get_tlink()

    def __get_ent_by_cols(self, cols: List[str]):
        if "SECTIME" not in self.__entity_df.columns:
            cols = [x for x in cols if x not in ["SECTIME", "SECTIME_context"]]
        return self.__entity_df[cols]

    def __get_tlink(self):
        return self.__relation_df

    def get_context_window(self, idx, text, tokens_expanded):
        # Get the flattened index of the current token
        token_start_idx = sum(len(str(t).split()) for t in text[:idx])

        # Extract the context window from the flattened list
        context_start = max(0, token_start_idx - self.window_size)
        context_end = min(len(tokens_expanded), token_start_idx + self.window_size)
        return " ".join(map(str, tokens_expanded[context_start:context_end]))
