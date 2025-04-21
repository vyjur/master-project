import configparser
import os
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import random
from structure.enum import (
    Dataset,
    Task,
    DocTimeRel,
    TLINK,
    SENTENCE,
    TAGS,
    TLINK_INPUT,
    ME,
    TIMEX,
)
from textmining.util import convert_to_input, is_date

import nltk

nltk.download("punkt")

random.seed(42)


class TRExtract:
    def __init__(
        self,
        config_file: str,
        manager: DatasetManager,
        task: Dataset,
        save_directory: str = "./src/textmining/tre/model",
        test_manager: DatasetManager = None,
    ):
        self.task = task
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)

        self.input_tag_type = self.__config["GENERAL"]["tag"]
        for tag_type in TAGS:
            if tag_type.name == self.input_tag_type:
                self.input_tag_type = tag_type
                break

        if task == Dataset.TLINK:
            self.tlink_input = self.__config["GENERAL"]["input"]
            for input in TLINK_INPUT:
                if input.name == self.tlink_input:
                    self.tlink_input = input
                    break

        load = self.__config["MODEL"].getboolean("load")
        print("LOAD", load)

        if self.__config.has_section("GENERAL") and self.__config.has_option(
            "GENERAL", "dct"
        ):
            self.__dct = self.__config.getboolean("GENERAL", "dct")
        else:
            self.__dct = False  # or whatever default you want

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.__config["pretrain"]["name"]
        )

        if self.__config.has_option("train.parameters", "downsample"):
            downsample = self.__config.getboolean("train.parameters", "downsample")
        else:
            downsample = False

        train_parameters = {
            "train_batch_size": self.__config.getint(
                "train.parameters", "train_batch_size"
            ),
            "valid_batch_size": self.__config.getint(
                "train.parameters", "valid_batch_size"
            ),
            "epochs": self.__config.getint("train.parameters", "epochs"),
            "learning_rate": self.__config.getfloat(
                "train.parameters", "learning_rate"
            ),
            "optimizer": self.__config["train.parameters"]["optimizer"],
            "weight_decay": self.__config.getfloat("train.parameters", "weight_decay"),
            "early_stopping_patience": self.__config.getint(
                "train.parameters", "early_stopping_patience"
            ),
            "early_stopping_delta": self.__config.getfloat(
                "train.parameters", "early_stopping_delta"
            ),
            "embedding_dim": self.__config.getint("train.parameters", "embedding_dim"),
            "shuffle": self.__config.getboolean("train.parameters", "shuffle"),
            "num_workers": self.__config.getint("train.parameters", "num_workers"),
            "max_length": self.__config.getint("train.parameters", "max_length"),
            "stride": self.__config.getint("train.parameters", "stride"),
            "weights": self.__config.getboolean("train.parameters", "weights"),
            "tune": self.__config.getboolean("tuning", "tune"),
            "tune_count": self.__config.getint("tuning", "count"),
            "downsample": downsample,
        }

        dataset = []
        tags = set()

        test_dataset = []

        if not load:
            dataset_tre = manager.get(task)
            if test_manager:
                test_dataset_tre = test_manager.get(task)

            if task == Dataset.DTR:
                dataset_tre = dataset_tre[dataset_tre["DCT"] != "BEFOREOVERLAP"]

                dataset, tags = self.__dtr(dataset_tre)

                if test_manager:
                    test_dataset_tre = test_dataset_tre[
                        test_dataset_tre["DCT"] != "BEFOREOVERLAP"
                    ]
                    test_dataset, _ = self.__dtr(test_dataset_tre)

            elif task == Dataset.TLINK:
                if test_manager:
                    print("Test set:")
                    test_dataset_tre = test_dataset_tre[
                        test_dataset_tre["RELATION"].notna()
                    ]

                    test_dataset_ner = test_manager.get(Dataset.NER)
                    test_dataset_ner = test_dataset_ner[
                        test_dataset_ner["MedicalEntity"].notna()
                    ].reset_index()
                    test_dataset_tee = test_manager.get(Dataset.TEE)

                    test_dataset, _ = self.__tlink(
                        test_dataset_tre, test_dataset_ner, test_dataset_tee
                    )

                print("Train dataset")
                dataset_ner = manager.get(Dataset.NER)
                dataset_ner = dataset_ner[
                    dataset_ner["MedicalEntity"].notna()
                ].reset_index()
                dataset_tee = manager.get(Dataset.TEE)
                # full_dataset = pd.concat([dataset_ner, dataset_tee])

                dataset, tags = self.__tlink(dataset_tre, dataset_ner, dataset_tee)

            tags = list(tags)
        else:
            if task == Dataset.DTR:
                tags = [cat.name for cat in DocTimeRel]
            else:
                tags = [cat.name for cat in TLINK]

        self.label2id, self.id2label = Util().get_tags(Task.SEQUENCE, tags)

        if task == Dataset.DTR and save_directory == "./src/textmining/tre/model":
            save_directory += "/dtr"
        elif task == Dataset.TLINK and save_directory == "./src/textmining/tre/model":
            save_directory += "/tlink"

        self.__model = MODEL_MAP[self.__config["MODEL"]["name"]](
            load,
            save_directory,
            dataset,
            tags,
            train_parameters,
            self.tokenizer,
            self.__config["GENERAL"]["name"],
            self.__config["pretrain"]["name"],
            testset=test_dataset,
        )

        self.preprocess = Preprocess(
            self.get_tokenizer(), self.get_max_length(), self.get_stride()
        )

    def __tlink(self, dataset_tre, dataset_ner, dataset_tee):
        tags = set()
        dataset = []
        for i, rel in dataset_tre.iterrows():
            if "ICD" in rel["FROM_CONTEXT"]:
                continue

            e_i = None
            e_j = None
            if dataset_ner is not None:
                res_e_i = dataset_ner[dataset_ner["Id"] == rel["FROM_Id"]]
                res_e_j = dataset_ner[dataset_ner["Id"] == rel["TO_Id"]]

                if len(res_e_i) > 0 and len(res_e_j) > 0:
                    e_i = res_e_i.iloc[0]
                    e_j = res_e_j.iloc[0]

                    if pd.isna(e_i["TIMEX"]) and pd.isna(e_j["TIMEX"]):
                        continue

            if e_i is None and e_j is None:
                e_i = {
                    "Text": rel["FROM"],
                    "Context": rel["FROM_CONTEXT"],
                }

                if is_date(e_i["Text"]):
                    e_i["TIMEX"] = True
                else:
                    e_i["MedicalEntity"] = True

                e_j = {
                    "Text": rel["TO"],
                    "Context": rel["TO_CONTEXT"],
                }
                if is_date(e_j["Text"]):
                    e_j["TIMEX"] = True
                else:
                    e_j["MedicalEntity"] = True

                if "TIMEX" in e_i and "TIMEX" in e_j:
                    continue

            if self.tlink_input == TLINK_INPUT.DIST:
                cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
            else:
                cat, distance = self.__class__.classify_tlink(e_i, e_j)

            text = self.classify_sep(e_i, e_j, cat, distance)

            relation = rel["RELATION"] if rel["RELATION"] != "BEFORE" else "O"
            # relation = rel["RELATION"]

            relation_pair = {"sentence": text, "relation": relation, "cat": cat}

            dataset.append(relation_pair)
            tags.add(relation)

        from collections import Counter

        neg_cp = []
        # Create negative candidate pairs
        for i, e_i in dataset_ner.iterrows():
            for j, e_j in dataset_tee.iterrows():
                if "ICD" in e_i["Context"] or e_i["Context"].strip() == "":
                    continue

                if (
                    str(e_i["Text"]) not in e_j["Context"]
                    and str(e_j["Text"]) not in e_i["Context"]
                ):
                    continue

                relations = dataset_tre[
                    (dataset_tre["FROM_Id"] == e_i["Id"])
                    & (dataset_tre["TO_Id"] == e_j["Id"])
                ]

                if len(relations) > 0:
                    continue
                else:
                    relation = "O"

                cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
                words = self.classify_sep(e_i, e_j, cat, distance)

                relation_pair = {"sentence": words, "relation": relation, "cat": cat}
                neg_cp.append(relation_pair)
                tags.add(relation)
        print("negative cp created:", len(neg_cp))
        dataset.extend(neg_cp)
        return dataset, tags

    def __dtr(self, dataset_tre):
        tags = set()
        dataset = []
        for _, row in dataset_tre.iterrows():
            if "ICD" in row["Context"]:
                continue
            if not self.__dct:
                dataset.append(
                    {
                        "sentence": convert_to_input(self.input_tag_type, row),
                        "relation": row["DCT"],
                    }
                )
            else:
                print("With DCT")
                e_i = {
                    "Text": row["SECTIME"],
                    "Context": row["SECTIME_context"],
                    "TIMEX": True,
                }

                e_j = {
                    "Text": row["Text"],
                    "Context": row["Context"],
                    "MedicalEntity": True,
                }

                sentence_i = convert_to_input(
                    self.input_tag_type, e_i, single=False, start=True
                )
                sentence_j = convert_to_input(
                    self.input_tag_type, e_j, single=False, start=False
                )
                text = f"{sentence_i} [SEP] {sentence_j}"
                dataset.append(
                    {
                        "sentence": text,
                        "relation": row["DCT"],
                    }
                )
            tags.add(row["DCT"])

        return dataset, tags

    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("train.parameters", "max_length")

    def get_stride(self):
        return self.__config.getint("train.parameters", "stride")

    def __run(self, data):
        output = self.__model.predict([val.ids for val in data])
        predictions = [self.id2label[i] for i in output[0]]
        return predictions, output[1]

    def batch_run(self, datas):
        batch_text = []
        for data in datas:
            e_j = None
            if type(data) is tuple:
                e_i = data[0]
                e_j = data[1]
            else:
                e_i = data

            e_i = {"Context": e_i["Context"], "Text": e_i["Text"]}

            if e_j is not None:
                e_j = {"Context": e_j["Context"], "Text": e_j["Text"]}

            if self.task == Dataset.DTR:
                text = convert_to_input(self.input_tag_type, e_i)
                batch_text.append(self.preprocess.run(text)[0])
            elif self.task == Dataset.TLINK:
                if e_j is None:
                    raise ValueError("Missing value for e_j")

                text = self.__process_tlink(e_i, e_j)
                batch_text.append(self.preprocess.run(text)[0])

        return self.__run(np.array(batch_text))

    def run(self, e_i, e_j=None):
        curr_e_i = {
            "Context": e_i.context,
            "Text": e_i.value,
            "MedicalEntity": e_i.type,
        }

        if isinstance(e_i.type, TIMEX):
            curr_e_i["TIMEX"] = e_i.type
        elif isinstance(e_i.type, ME):
            curr_e_i["MedicalEntity"] = e_i.type

        if e_j is not None:
            curr_e_j = {
                "Context": e_j.context,
                "Text": e_j.value,
            }
            if isinstance(e_j.type, TIMEX):
                curr_e_j["TIMEX"] = e_j.type
            elif isinstance(e_i.type, ME):
                curr_e_j["MedicalEntity"] = e_j.type

        if self.task == Dataset.DTR:
            text = convert_to_input(self.input_tag_type, curr_e_i)
        else:
            if e_j is None:
                raise ValueError("Missing value for e_j")

            text = self.__process_tlink(curr_e_i, curr_e_j)

        print(text)

        output = self.__run(self.preprocess.run(text))
        return output[0][0], output[1]

    def __process_tlink(self, e_i, e_j):
        cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
        text = self.classify_sep(e_i, e_j, cat, distance)
        return text

    def classify_sep(self, e_i, e_j, cat, distance):
        sentence_i = convert_to_input(
            self.input_tag_type, e_i, single=False, start=True
        )
        # TODO:
        # if cat == SENTENCE.INTRA:
        e_j["Context"] = sentence_i

        sentence_j = convert_to_input(
            self.input_tag_type, e_j, single=False, start=False
        )
        text = sentence_j
        # TODO:
        # if cat == SENTENCE.INTRA:
        #     text = sentence_j
        # else:
        #     if self.tlink_input == TLINK_INPUT.DIST:
        #         text = f"{sentence_i} [SEP_{distance}] {sentence_j}"
        #     else:
        #         text = f"{sentence_i} [SEP] {sentence_j}"
        return text

    @staticmethod
    def classify_tlink(e_i, e_j, distance=False):
        sentences = sent_tokenize(e_i["Context"])

        entity1_index = entity2_index = None
        cat = SENTENCE.INTER

        for i, sentence in enumerate(sentences):
            # It is inter sentence if both entities are in the same sentence
            if str(e_i["Text"]) in sentence and str(e_j["Text"]) in sentence:
                if not distance:
                    return SENTENCE.INTRA, 100
                else:
                    cat = SENTENCE.INTRA

            if distance:
                if str(e_i["Text"]) in sentence and entity1_index is None:
                    entity1_index = i
                if str(e_j["Text"]) in sentence and entity2_index is None:
                    entity2_index = i
        if distance:
            if entity1_index is not None and entity2_index is not None:
                return cat, abs(entity2_index - entity1_index) - 1
            else:
                return cat, 100

        return cat, 100


if __name__ == "__main__":
    from structure.enum import Dataset
    from structure.node import Node

    folder_path = "./data/annotated_MTSamples"
    files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    manager = DatasetManager(files)

    reg = TRExtract("./src/textmining/tre/config.ini", manager, Dataset.DTR)

    e_i = Node("tungpust", None, None, "Han har tungpust", None)
    e_j = Node("brystsmerter", None, None, "Brystsmertene har vart en stund.", None)

    print("Result:", reg.run(e_i, e_j))
