import configparser
import os
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from structure.enum import Dataset, Task, ME, NER_SCHEMA


class MERecognition:
    def __init__(
        self,
        config_file: str,
        manager: DatasetManager,
        save_directory: str = "./models/ner",
        test_manager: DatasetManager = None,
    ):
        self.__config = configparser.ConfigParser(allow_no_value=True)
        self.__config.read(config_file)

        load = self.__config["MODEL"].getboolean("load")
        print("LOAD", load)

        dataset = []
        test_dataset = []

        self.schema = self.__config["MODEL"]["schema"]

        for sch in NER_SCHEMA:
            if self.schema == sch.name:
                self.schema = sch
                break

        if load:
            tags = [cat.name for cat in ME]
        else:
            dataset = manager.get(Dataset.NER)
            dataset["MedicalEntity"] = dataset["MedicalEntity"].fillna("O")

            if test_manager:
                test_dataset = test_manager.get(Dataset.NER)
                test_dataset["MedicalEntity"] = test_dataset["MedicalEntity"].fillna(
                    "O"
                )
                test_dataset = [test_dataset]
            tags = dataset["MedicalEntity"].unique()
            dataset = [dataset]
            tags = list(tags)

        self.__util = Util(schema=self.schema)
        self.label2id, self.id2label = self.__util.get_tags(Task.TOKEN, tags)

        print(self.label2id, self.id2label)

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

        self.__model = MODEL_MAP[self.__config["MODEL"]["name"]](
            load,
            save_directory,
            dataset,
            tags,
            parameters=train_parameters,
            tokenizer=self.tokenizer,
            project_name=self.__config["GENERAL"]["name"],
            pretrain=self.__config["pretrain"]["name"],
            util=self.__util,
            testset=test_dataset,
        )

    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("train.parameters", "max_length")

    def get_stride(self):
        return self.__config.getint("train.parameters", "stride")

    def get_util(self):
        return self.__util

    def run(self, data):
        output, _ = self.__model.predict([val.ids for val in data])
        predictions = [[self.id2label[int(j.cpu().numpy())] for j in i] for i in output]
        return predictions

    def get_model(self):
        return self.__model

    def get_non_o_intervals(self, lst):
        intervals = []
        start = None

        prev_value = "O"
        prev_orig = "O"

        for i, value in enumerate(lst):
            cat_value = self.__util.remove_schema(value)

            if value == "O":
                if start is not None:
                    intervals.append((start, i))
                    start = None

            if self.schema == NER_SCHEMA.BIO:
                if (
                    value.startswith("B-")
                    or (value.startswith("I-") and start is None)
                    or cat_value != prev_value
                ):
                    if start is not None:
                        intervals.append((start, i))
                    if value != "O":
                        start = i

            elif self.schema == NER_SCHEMA.IO:
                if value.startswith("I-") and (
                    start is None or cat_value != prev_value
                ):
                    if start is not None and cat_value != prev_value:
                        intervals.append((start, i))
                    start = i

            elif self.schema == NER_SCHEMA.IOE:
                if value.startswith("I-") or value.startswith("E-"):
                    if start is None:
                        start = i
                    elif cat_value != prev_value or (
                        prev_orig.startswith("E-") and value.startswith("I-")
                    ):
                        intervals.append((start, i))
                        start = i
                elif value.startswith("E-") and start is not None:
                    intervals.append((start, i + 1))
                    start = None
            else:
                raise TypeError("No schema selected.")

            prev_value = cat_value
            prev_orig = value

        # If the last element is part of an interval
        if start is not None:
            intervals.append((start, len(lst)))

        return intervals


if __name__ == "__main__":
    folder_path = "./data/annotated/"
    files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    manager = DatasetManager(["./data/annotated_MTSamples/ex13.tsv"])

    reg = MERecognition("./src/textmining/ner/config.ini", manager)
    preprocess = Preprocess(
        reg.get_tokenizer(), reg.get_max_length(), reg.get_stride(), reg.get_util()
    )

    text = "Pasienten har også opplevd økt tungpust de siste månedene, noe som har begrenset aktivitetsnivået hans."

    print(reg.run(preprocess.run(text)))
