import configparser
import os
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import random
from structure.enum import Dataset, Task, DocTimeRel, TLINK, SENTENCE

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


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
            "optimizer": self.__config["train.parameters"]["optimizer"],
            "weight_decay": self.__config.getfloat("train.parameters", "weight_decay"),
            "early_stopping_patience": self.__config.getint("train.parameters", "early_stopping_patience"),
            "early_stopping_delta": self.__config.getfloat("train.parameters", "early_stopping_delta"),
            "embedding_dim": self.__config.getint("train.parameters", "embedding_dim"),
            "shuffle": self.__config.getboolean("train.parameters", "shuffle"),
            "num_workers": self.__config.getint("train.parameters", "num_workers"),
            "max_length": self.__config.getint("train.parameters", "max_length"),
            "stride": self.__config.getint("train.parameters", "stride"),
            "tune": self.__config.getboolean("tuning", "tune"),
            "tune_count": self.__config.getint("tuning", "count") 
        }

        dataset = []
        tags = set()
        
        if not load:
            dataset_ner = manager.get(Dataset.NER)
            dataset_ner = dataset_ner[dataset_ner['MedicalEntity'].notna() | dataset_ner['TIMEX'].notna()].reset_index()
            dataset_tre = manager.get(task)
            if task == Dataset.DTR:
                for _, row in dataset_tre.iterrows():
                    dataset.append(
                        {
                            "sentence": row['Context']
                            .replace(row['Text'], f"<TAG>{row['Text']}</TAG>"),
                            "relation": row['DCT'],
                        }
                    )
                    tags.add(row['DCT'])
            elif task == Dataset.TLINK:
                
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
                    
                    cat = self.classify_tlink(e_i, e_j)
                    
                    relation_pair = {
                        "sentence": words,
                        "relation": rel["RELATION"],
                        "cat": cat
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

                        cat = self.__class__.classify_tlink(e_i, e_j)
                        relation_pair = {
                            "sentence": words,
                            "relation": relation,
                            "cat": cat
                        }
                        dataset.append(relation_pair)
                        tags.add(relation)

            tags = list(tags)
        else:
            if task == Dataset.DTR:
                tags = [cat.name for cat in DocTimeRel]
            else:
                tags = [cat.name for cat in TLINK]
                
        self.label2id, self.id2label = Util().get_tags(
            Task.SEQUENCE, tags
        )

        if task == Dataset.DTR and save_directory == "./src/textmining/tre/model":
            save_directory += "/dtr"
        elif (
            task == Dataset.TLINK and save_directory == "./src/textmining/tre/model"
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

        self.preprocess = Preprocess(self.get_tokenizer(), self.get_max_length(), self.get_stride())

    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("MODEL", "max_length")
    
    def get_stride(self):
        return self.__config.getint("MODEL", "stride")

    def __run(self, data):
        output = self.__model.predict([val.ids for val in data])
        predictions = [self.id2label[i] for i in output[0]]
        return predictions[0], output[1]

    def run(self, e_i, e_j=None):
        if self.task == Dataset.DTR:
            text = e_i.context.replace(e_i.value, f"<TAG>{e_i.value}</TAG>")
        else:
            if e_j is None:
                raise ValueError("Missing value for e_j")
            sentence_i = e_i.context.replace(e_i.value, f"<TAG>{e_i.value}</TAG>")
            sentence_j = e_j.context.replace(e_j.value, f"<TAG>{e_j.value}</TAG>")
            text = f"{sentence_i} [SEP] {sentence_j}"
        return self.__run(self.preprocess.run(text))
   
    @static 
    def classify_tlink(self, e_i, e_j):
        sentences = sent_tokenize(e_i['Context'])
        
        for sentence in sentences:
            # It is inter sentence if both entities are in the same sentence
            if e_i['Text'] in sentence and e_j['Text'] in sentence:
                return SENTENCE.INTRA
        return SENTENCE.INTER


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
