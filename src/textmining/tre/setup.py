import configparser
import os
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import random
from structure.enum import Dataset, Task, TR_DCT, TR_TLINK


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
        dataset_ner = dataset_ner[dataset_ner['MedicalEntity'].notna() | dataset_ner['TIMEX'].notna()].reset_index()
        dataset_tre = manager.get(task)
        
        dataset = []
        tags = set()
        
        if not load:
            if task == Dataset.TRE_DCT:
                for _, row in dataset_tre.iterrows():
                    dataset.append(
                        {
                            "sentence": row['Context']
                            .replace(row['Text'], f"<TAG>{row['Text']}</TAG>"),
                            "relation": row['DCT'],
                        }
                    )
                    tags.add(row['DCT'])
            elif task == Dataset.TRE_TLINK:
                
                for i, rel in dataset_tre.iterrows():
                    
                    e_i = dataset_ner[dataset_ner['Id'] == rel['FROM_Id']]
                    e_j = dataset_ner[dataset_ner['Id'] == rel['TO_Id']]
                    
                    if len(e_i) > 0 and len(e_j) > 0:
                        e_i = e_i.iloc[0]
                        e_j = e_j.iloc[0]
                    else:
                        continue
                
                    sentence_i = e_i['Context'].replace(e_i['Text'], f"<TAG>{e_i['Text']}</TAG>")
                    sentence_j = e_j['Context'].replace(e_j['Text'], f"<TAG>{e_j['Text']}</TAG>")
                    
                    words = f"{sentence_i} [SEP] {sentence_j}"

                    relation_pair = {
                        "sentence": words,
                        "relation": rel["RELATION"],
                    }
                    dataset.append(relation_pair)
                    tags.add(rel["RELATION"])
                    
                for i, e_i in dataset_ner.iterrows():
                    for j, e_j in dataset_ner.iterrows():
                        if i == j:
                            continue
                        
                        if e_i['Text'] not in e_j['Context'] and e_j['Text'] not in e_i['Context']:
                            continue

                        relations = dataset_tre[
                            (dataset_tre["FROM_Id"] == e_i['Id'])
                            & (dataset_tre["TO_Id"] == e_j['Id'])
                        ]

                        if len(relations) > 0:
                            continue
                        else:
                            relation = "O"

                            # TODO: downsample majority class
                            if random.random() < 0.999:
                                continue

                        # TODO: Add amount of SEP as sentences between them?
                        sentence_i = e_i['Context'].replace(e_i['Text'], f"<TAG>{e_i['Text']}</TAG>")
                        sentence_j = e_j['Context'].replace(e_j['Text'], f"<TAG>{e_j['Text']}</TAG>")
                        
                        words = f"{sentence_i} [SEP] {sentence_j}"

                        relation_pair = {
                            "sentence": words,
                            "relation": relation,
                        }
                        dataset.append(relation_pair)
                        tags.add(relation)

            tags = list(tags)
        else:
            if task == Dataset.TRE_DCT:
                tags = [cat.name for cat in TR_DCT]
            else:
                tags = [cat.name for cat in TR_TLINK]
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
