import configparser
import os
import numpy as np
from types import SimpleNamespace
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import random
from structure.enum import Dataset, Task, DocTimeRel, TLINK, SENTENCE, TAGS, TLINK_INPUT
from textmining.util import convert_to_input

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
random.seed(42)


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
        
        self.input_tag_type = self.__config['GENERAL']['tag']
        for tag_type in TAGS:
            if tag_type.name == self.input_tag_type:
                self.input_tag_type = tag_type
                break
        
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
            "weights": self.__config.getboolean("train.parameters", "weights"),
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
                dataset_tre = dataset_tre[dataset_tre['DCT'] != 'BEFOREOVERLAP']
                for _, row in dataset_tre.iterrows():
                    if 'ICD' in row['Context']:
                        continue
                    dataset.append(
                        {
                            "sentence": convert_to_input(self.input_tag_type, row),
                            "relation": row['DCT'],
                        }
                    )
                    tags.add(row['DCT'])
            elif task == Dataset.TLINK:
                
                self.tlink_input = self.__config['GENERAL']['input']
                for input in TLINK_INPUT:
                    if input.name == self.tlink_input:
                        self.tlink_input = input
                        break
                
                for i, rel in dataset_tre.iterrows():
                    
                    if rel['FROM_CONTEXT'].strip() == "" or rel['TO_CONTEXT'].strip() == "" or 'ICD' in rel['FROM_CONTEXT']:
                        e_i = dataset_ner[dataset_ner['Id'] == rel['FROM_Id']]
                        e_j = dataset_ner[dataset_ner['Id'] == rel['TO_Id']]
                    
                        if len(e_i) > 0 and len(e_j) > 0:
                            e_i = e_i.iloc[0]
                            e_j = e_j.iloc[0]
                        else:
                            continue
                    
                    if self.tlink_input == TLINK_INPUT.DIST:
                        cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
                    else:
                        cat = self.__class__.classify_tlink(e_i, e_j)
                    sentence_i = convert_to_input(self.input_tag_type, e_i, single=False, start=True)
                    if cat == SENTENCE.INTRA:
                        e_j['Context'] = sentence_i
                    
                    sentence_j = convert_to_input(self.input_tag_type, e_j, single=False, start=False)
                    
                    if cat == SENTENCE.INTRA:
                        text = sentence_j
                    else:
                        if self.tlink_input == TLINK_INPUT.DIST:
                            text = f"{sentence_i} [SEP_{distance}] {sentence_j}"
                        else:
                            text = f"{sentence_i} [SEP] {sentence_j}"
                        
                                        
                    relation_pair = {
                        "sentence": text,
                        "relation": rel["RELATION"],
                        "cat": cat
                    }
                        
                    dataset.append(relation_pair)
                    tags.add(rel["RELATION"])
                
                # Create negative candidate pairs
                for i, e_i in dataset_ner.iterrows():
                    for j, e_j in dataset_ner.iterrows():
                        if i == j or 'ICD' in e_i['Context'] or e_i['Context'].strip() == '':
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

                            # TODO: downsample majority class: this is not deterministic
                            if random.random() < 0.5:
                                continue

                        sentence_i = convert_to_input(self.input_tag_type, e_i, single=False, start=True)
                        sentence_j = convert_to_input(self.input_tag_type, e_j, single=False, start=False)
                        
                        cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
                        words = self.classify_sep(e_i, e_j, cat, distance)

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
            if type(data) is list:
                e_i = data[0]
                e_j = data[1]
            else:
                e_i = data
                
            e_i = {
                "Context": e_i.context,
                "Text": e_i.value
            }
            
            if e_j is not None:
                e_j = {
                    "Context": e_j.context,
                    "Text": e_j.value
                }
            
            if self.task == Dataset.DTR:
                text = convert_to_input(self.input_tag_type, e_i)
                batch_text.append(self.preprocess.run(text)[0])
            elif self.task == Dataset.TLINK:
                if e_j is None:
                    raise ValueError("Missing value for e_j")
                cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
                text = self.classify_sep(e_i, e_j, cat, distance)
                batch_text.append(self.preprocess.run(text)[0])
                
        return self.__run(np.array(batch_text))
    
    def run(self, e_i, e_j=None):
        
        e_i = {
            "Context": e_i.context,
            "Text": e_i.value
        }
        
        if e_j is not None:
            e_j = {
                "Context": e_j.context,
                "Text": e_j.value
            }
            
        if self.task == Dataset.DTR:
            text = convert_to_input(self.input_tag_type, e_i)
        else:
            if e_j is None:
                raise ValueError("Missing value for e_j")
            cat, distance = self.__class__.classify_tlink(e_i, e_j, True)
            text = self.classify_sep(e_i, e_j, cat, distance)
                        
        output = self.__run(self.preprocess.run(text))
        return output[0][0], output[1]
    
    def classify_sep(self, e_i, e_j, cat, distance):
        sentence_i = convert_to_input(self.input_tag_type, e_i, single=False, start=True)
        if cat == SENTENCE.INTRA:
            e_j['Context'] = sentence_i
        
        sentence_j = convert_to_input(self.input_tag_type, e_j, single=False, start=False)
        
        if cat == SENTENCE.INTRA:
            text = sentence_j
        else:
            if self.tlink_input == TLINK_INPUT.DIST:
                text = f"{sentence_i} [SEP_{distance}] {sentence_j}"
            else:
                text = f"{sentence_i} [SEP] {sentence_j}"
        return text
   
    @staticmethod
    def classify_tlink(e_i, e_j, distance=False):
        sentences = sent_tokenize(e_i['Context'])
        
        entity1_index = entity2_index = None
        cat = SENTENCE.INTER
        
        for i, sentence in enumerate(sentences):
            # It is inter sentence if both entities are in the same sentence
            if e_i['Text'] in sentence and e_j['Text'] in sentence:
                if not distance:
                    return SENTENCE.INTRA
                else:
                    cat = SENTENCE.INTRA
                  
            if distance:  
                if e_i['Text'] in sentence and entity1_index is None:
                    entity1_index = i
                if e_j['Text'] in sentence and entity2_index is None:
                    entity2_index = i
        if distance:
            if entity1_index is not None and entity2_index is not None:
                return cat, abs(entity2_index - entity1_index) - 1
            else:
                return cat, 100    
                    
        return cat


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
