import configparser
import os
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import random
from structure.enum import Dataset, Task


class TRExtract:
    def __init__(
        self,
        config_file: str,
        manager: DatasetManager,
        task: Dataset,
        save_directory: str = "./src/textmining/tre/model",
    ):
        self.task = task
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)

        load = self.__config["MODEL"].getboolean("load")
        print("LOAD", load)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.__config["pretrain"]["name"]
        )

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
            "shuffle": self.__config.getboolean("train.parameters", "shuffle"),
            "num_workers": self.__config.getint("train.parameters", "num_workers"),
            "max_length": self.__config.getint("MODEL", "max_length"),
        }

        dataset_ner = manager.get(Dataset.NER)
        dataset_tre = manager.get(task)
        sentences = manager.get(Dataset.SENTENCES)

        dataset = []
        tags = set()
        for k, doc in enumerate(dataset_ner):
            for i, e_i in enumerate(doc.itertuples()):
                if task == Dataset.TRE_TLINK:
                    for j, e_j in enumerate(doc.itertuples()):
                        if i == j:
                            continue
                        relations = dataset_tre[k][
                            (dataset_tre[k]["fk_id"] == e_i[1])
                            & (dataset_tre[k]["id"] == e_j[1])
                        ]

                        if len(relations) == 1:
                            relation = relations.iloc[0]["TRE_TLINK"]
                        else:
                            relation = "O"

                            # TODO: downsample majority class
                            if random.random() < 0.999:
                                continue

                        # TODO: change setup of input with XML tags? add amount of SEP as sentences between them?
                        sentence_i = sentences[k].loc[e_i[2]].replace(e_i[3], f"<TAG>{e_i[3]}</TAG>")
                        sentence_j = sentences[k].loc[e_j[2]].replace(e_j[3], f"<TAG>{e_j[3]}</TAG>")
                        
                        words = f"{sentence_i} [SEP] {sentence_j}"

                        relation_pair = {
                            "sentence": words,
                            "relation": relation,
                        }
                        dataset.append(relation_pair)
                        tags.add(relation)
                else:
                    ### Info: This is for TRE_DCT
                    dct = list(
                        set(dataset_tre[k][dataset_tre[k]["id"] == e_i[1]]["TRE_DCT"])
                    )
                    if len(dct) > 0:
                        dct = dct[0]
                    else:
                        continue
                    dataset.append(
                        {
                            "sentence": sentences[k]
                            .loc[e_i[2]]
                            .replace(e_i[3], f"<TAG>{e_i[3]}</TAG>"),
                            "relation": dct,
                        }
                    )
                    tags.add(dct)

        tags = list(tags)
        self.label2id, self.id2label = Util().get_tags(
            Task.SEQUENCE, tags, task != Dataset.TRE_DCT
        )

        if task == Dataset.TRE_DCT and save_directory == "./src/textmining/tre/model":
            save_directory += "/dct"
        elif (
            task == Dataset.TRE_TLINK and save_directory == "./src/textmining/tre/model"
        ):
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
        )

        self.preprocess = Preprocess(self.get_tokenizer(), self.get_max_length())

    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("MODEL", "max_length")

    def __run(self, data):
        # TODO: fix here
        output = self.__model.predict([val.ids for val in data])
        predictions = [self.id2label[i] for i in output[0]]
        return predictions[0], output[1]

    def run(self, e_i, e_j=None):
        if self.task == Dataset.TRE_DCT:
            text = e_i.context.replace(e_i.value, f"<TAG>{e_i.value}</TAG>")
        else:
            if e_j is None:
                raise ValueError("Missing value for e_j")
            sentence_i = e_i.context.replace(e_i.value, f"<TAG>{e_i.value}</TAG>")
            sentence_j = e_j.context.replace(e_j.value, f"<TAG>{e_j.value}</TAG>")
            text = f"{sentence_i} [SEP] {sentence_j}"
        return self.__run(self.preprocess.run(text))


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

    reg = TRExtract("./src/textmining/tre/config.ini", manager, Dataset.TRE_DCT)

    e_i = Node("tungpust", None, None, "Han har tungpust", None)
    e_j = Node("brystsmerter", None, None, "Brystsmertene har vart en stund.", None)

    print("Result:", reg.run(e_i, e_j))
