import pandas as pd
from typing import List
from structure.enum import Dataset

### Info: This is old version in annotated folder
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

COLUMN_NAMES = [
    "id", # 0,
    "sentence_id"
    "Offset",  # 1
    "Text",  # 2
    "MER",  # 4 ,
    "TRE_DCT",  # 3,
    "TRE_TLINK",  # 5,
    "fk_id",  # 6
]


class DatasetManager:
    def __init__(self, files: List[str]):
        self.__documents = []
        for file in files:
            document = pd.DataFrame(columns=COLUMN_NAMES)  # type: ignore

            with open(file, encoding="UTF-8") as f:
                connect = False
                prev_id = -1
                for i, line in enumerate(f):
                    if i in range(6) or line.strip() in ["\n", ""]:
                        continue

                    if line.startswith("#"):
                        continue
                    sentence = line.split("\t")

                    fk_id = sentence[6]
                    clip = fk_id.find("[")
                    if clip != -1:
                        fk_id = fk_id[:clip]

                    fk_id = fk_id.split("|")
                    t_relation = sentence[5].split("|")

                    clip = sentence[4].find("[")
                    if clip == -1:
                        clipper = len(sentence[4])
                    else:
                        clipper = clip

                    if not connect and clip != -1:
                        connect = True
                        prev_id = sentence[0]
                    elif clip == -1:
                        connect = False

                    if connect and prev_id != sentence[0]:
                        for i in range(len(document)):
                            if document.loc[i]["id"] == prev_id:
                                document.loc[i, "Text"] = (
                                    document.loc[i, "Text"] + " " + sentence[2]
                                )
                    else:
                        for i, id in enumerate(fk_id):
                            row = {
                                "id": sentence[0] if sentence[0] != "_" else "O",  # 0
                                "sentence_id": int(sentence[0].split("-")[0]),
                                "Offset": sentence[1]
                                if sentence[1] != "_"
                                else "O",  # 1
                                "Text": sentence[2] if sentence[2] != "_" else "O",  # 1
                                "MER": sentence[4][:clipper]
                                if sentence[4j] != "_"
                                else "O",  # 4
                                "TRE_DCT": sentence[3]
                                if sentence[3] != "_"
                                else "O",  # 3,
                                "TLINK": t_relation[i]
                                if t_relation[i] != "_"
                                else None,  # 13,
                                "fk_id": id if id != "_" else None,  # 14
                            }
                            document.loc[len(document)] = row

            self.__documents.append(document)

    def get(self, task: Dataset):
        match task:
            case Dataset.NER:
                return self.__get_docs_by_cols(
                    ["id", "sentence_id", "Text", "MER"]
                )
            case Dataset.TRE_DCT:
                return self.__get_docs_by_cols(
                    ["id", "Text", "TRE_DCT"]
                )
            case Dataset.TRE_TLINK:
                return self.__get_docs_by_cols(
                    ["id", "Text", "TRE_TLINK", "fk_id"]
                )
            case Dataset.SENTENCES:
                return self.__get_sentences()
            case _:
                return None

    def __get_docs_by_cols(self, cols: List[str]):
        return [doc[cols].drop_duplicates() for doc in self.__documents]

    def __get_sentences(self):
        return [
            doc.groupby("sentence_id")["Text"].apply(lambda x: " ".join(x))
            for doc in self.__documents
        ]


if __name__ == "__main__":
    manager = DatasetManager(
        ["./data/annotated/journal.tsv", "./data/annotated/journal-2.tsv"]
    )

    print(manager.get(Dataset.TRE_DCT))

