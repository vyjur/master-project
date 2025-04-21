from textmining.tee.heideltime.python_heideltime.main import Heideltime
from textmining.tee.rules import *
import xml.etree.ElementTree as ET
from preprocess.dataset import DatasetManager
import configparser
from structure.enum import Task, Dataset, DCT, TAGS
from model.util import Util
import pandas as pd
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from model.map import MODEL_MAP
from textmining.util import convert_to_input
import html
import numpy as np



class TEExtract:
    
    def __init__(self, config_file:str, manager: DatasetManager, save_directory: str = "./src/textmining/tee/model", rules:bool=True, test_manager:DatasetManager = None):
        self.__heideltime = Heideltime()
        self.__heideltime.set_document_type('NEWS')
        self.__heideltime.set_language('auto-norwegian')
        self.__rules = rules
        
        self.__config = configparser.ConfigParser(allow_no_value=True)
        self.__config.read(config_file)
        
        self.input_tag_type = self.__config["GENERAL"]['tag']
        
        for tag_type in TAGS:
            if tag_type.name == self.input_tag_type:
                self.input_tag_type = tag_type
                break
        
        load = self.__config["MODEL"].getboolean("load")
        print("LOAD", load)
        
        dataset = []
        test_dataset = []
        tags = [DCT.DATE.name, DCT.DCT.name]   
        if not load:
            tags = set()
            raw_dataset = manager.get(Dataset.TEE)
            for _, row in raw_dataset.iterrows():
                if row['Text'].replace(" ", "").isalpha() or 'ICD' in row['Context']:
                    continue
                dataset.append(
                    {
                        "sentence": convert_to_input(self.input_tag_type, row, True, True),
                        "relation": row['TIMEX'],
                    }
                )
                tags.add(row['TIMEX'])
            if test_manager:
                raw_test_dataset = test_manager.get(Dataset.TEE)
                for _, row in raw_test_dataset.iterrows():
                    if row['Text'].replace(" ", "").isalpha() or 'ICD' in row['Context']:
                        continue
                    test_dataset.append(
                        {
                            "sentence": convert_to_input(self.input_tag_type, row, True, True),
                            "relation": row['TIMEX'],
                        }
                    )
            tags = list(tags)
        self.label2id, self.id2label = Util().get_tags(Task.SEQUENCE, tags)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.__config["pretrain"]["name"]
        )
        
        if self.__config.has_option('train.parameters', "downsample"):
            downsample = self.__config.getboolean('train.parameters', 'downsample')
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
            "early_stopping_patience": self.__config.getint("train.parameters", "early_stopping_patience"),
            "early_stopping_delta": self.__config.getfloat("train.parameters", "early_stopping_delta"),
            "embedding_dim": self.__config.getint("train.parameters", "embedding_dim"),
            "shuffle": self.__config.getboolean("train.parameters", "shuffle"),
            "num_workers": self.__config.getint("train.parameters", "num_workers"),
            "max_length": self.__config.getint("train.parameters", "max_length"),
            "stride": self.__config.getint("train.parameters", "stride"),
            "weights": self.__config.getboolean("train.parameters", "weights"),
            "tune": self.__config.getboolean("tuning", "tune"),
            "tune_count": self.__config.getint("tuning", "count"),
            "downsample": downsample
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
            testset=test_dataset
        )
        
        self.preprocess = Preprocess(self.get_tokenizer(), self.get_max_length(), self.get_stride())
        
    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("train.parameters", "max_length")

    def get_stride(self):
        return self.__config.getint("train.parameters", "stride")

    def __model_run(self, data, prob=False):
        output, prob = self.__model.predict([val.ids for val in data])
        predictions = [self.id2label[int(i)] if int(i) in self.id2label else "O" for i in output ]
        if prob:
            return predictions, prob
        return predictions

    def get_model(self):
        return self.__model
        
    def set_dct(self, dct):
        self.__heideltime.set_document_time(dct)
        
    def pre_rules(self, text, sectime=False):
        text = convert_date_format_3(text)
       
        if not sectime:
            
            text = convert_text(text)
            text = convert_negative_years(text)
            text = convert_full_year(text)
            
        text = expand_norwegian_months(text)
        text = convert_date_format_2(text)
        text = convert_date_format(text)
        text = convert_slash_date(text)
        
        if self.__heideltime.document_time is not None and not sectime:
            # Rule 4: 25.12 => 25.12.YYYY where YYYY is the same year as DCT if 25.12.YYYY < DCT. Else, the year before that.
            dct = datetime.strptime(self.__heideltime.document_time, "%Y-%m-%d")
            
            try:
                text = convert_partial_dates(text, dct)
            except:
                pass
        return text
            
    def __post_rules(self, full_text, text, ttype, value):
        check_context = self.__get_window(full_text, text, window_size=3)

        if self.__heideltime.document_time is not None:
            dct = datetime.strptime(self.__heideltime.document_time, "%Y-%m-%d")
            if ttype == "DURATION":
                return convert_duration(check_context, value, dct), True
        return value, False
        
    def run(self, text, sectime=False):
        if self.__rules:
            text = self.pre_rules(text, sectime)
    
        result = self.__heideltime.parse(text).encode("utf-8").decode("utf-8")
        try:
            root = ET.fromstring(result)
        except ET.ParseError as e:
            print(f"XML parsing error: {e}")
            columns = ["id", "type", "value", "text", "context"]
            return pd.DataFrame(columns=columns)
        full_text = " ".join(root.itertext())

        timex_elements = root.findall('.//TIMEX3')
        
        data = []
        for timex in timex_elements:
            tid = timex.get('tid')
            ttype = timex.get('type')
            value = timex.get('value')
            text = timex.text
            
            if text.replace(' ', '').isalpha():
                continue
            
            # Get the full text of the document
            
            # Get the 50-token window around this TIMEX3 element
            context = self.__get_window(full_text, text, window_size=50)
            
            if self.__rules and not sectime: 
                value, changed = self.__post_rules(full_text, text, ttype, value)
                ttype = "DATE" if changed else ttype
            
            data.append( {
                'id': tid,
                'type': ttype,
                'value': value,
                'text': text,
                'context': context
            })
            
        if not data:  # This checks if the list is empty
            return pd.DataFrame(columns=['id', 'type', 'value', 'text', 'context'])
        
        output = pd.DataFrame(data)
        print(output)
        return output[output['type'] == 'DATE']
            
    def extract_sectime(self, data):
        # Initial output: Extracting all TIMEX expressions
        init_output = self.run(data, True)
        data = self.pre_rules(data, True)
        
        # Only DATE expressions are candidate for DCT 
        dct_candidates = init_output[init_output['type'] == 'DATE']
        
        # For each candidate classify if it is really a DCT /SECTIME or not
        sections = []
        dcts = []

        for i, row in dct_candidates.iterrows():
            if row['text'].replace(" ", "").isalpha():
                continue
            dct_output = self.predict_sectime(row)
            if dct_output[0][0] == "DCT":
                dcts.append(row)
                dct_start = data.index(row['text'])
                start = dct_start + len(row['text'])
                sections.append({
                    "index": start,
                    "value": row['value']    
                })
            
        return dcts, sections
    
    def predict_sectime(self, data, prob=False):
        if data['text'].isalpha():
            return 'DATE'
        data = {
            'Context': data['context'],
            'Text': data['text'],
            "TIMEX": True
        }
        text = convert_to_input(self.input_tag_type, data, True, True)
        print(f"Value: {data['Text']} Context: {text}")
        return self.__model_run(self.preprocess.run(text), prob)
    
    def batch_predict_sectime(self, datas, prob=False):
        batch_text = []
        for data in datas:
            data = {
                'Context': data['context'],
                'Text': data['text']
            }
            text = convert_to_input(self.input_tag_type, data, True, True)
            batch_text.append(self.preprocess.run(text)[0])
        
        if len(batch_text) == 0:
            return None
        return self.__model_run(np.array(batch_text), prob)

    def __get_window(self, full_text, timex_text, window_size=50):
        # Find the index of the TIMEX3 text in the full text
        start_index = full_text.find(timex_text)
        end_index = start_index + len(timex_text)
        
        # Get the character window before and after the TIMEX3 text
        before_window = full_text[:start_index].split() # 50 characters before TIMEX3
        before_window = before_window[max(0, len(before_window) - window_size):]
        after_window = full_text[end_index:].split()  # 50 characters after TIMEX3
        after_window = after_window[:min(len(after_window), window_size)]
        
        # Combine the before and after windows with the TIMEX3 text itself in the middle
        window = " ".join(before_window) + " " + timex_text + " " + " ".join(after_window)
        return window
        
        
if __name__ == '__main__':
    import os
    from structure.enum import Dataset
    from preprocess.dataset import DatasetManager
    from sklearn.metrics import classification_report
    
    folder_path = "./data/helsearkiv/annotated/entity/"

    entity_files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    folder_path = "./data/helsearkiv/annotated/relation/"

    relation_files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))

    ]

    tee = TEExtract()
    tee.set_dct('2025-02-10')
     
    # manager = DatasetManager(entity_files, relation_files, False)
    
    # dataset = manager.get(Dataset.TEE)
    
    # print(dataset)

    # target = []
    # pred = []
    # for i, data in dataset.iterrows():
    #     output = tee.run([data['Text']])[0]
    #     if not output.empty:
    #         if output["type"].values[0] != data['TIMEX'].replace('DCT', 'DATE'):
    #             print(data['Text'], output["type"].values[0], data['TIMEX'].replace('DCT', 'DATE'), data['Context'])
    #         pred.append(output["type"].values[0])
    #         target.append(data['TIMEX'].replace('DCT', 'DATE'))
    #     else:
    #         pred.append('O')
    #         target.append(data['TIMEX'].replace('DCT', 'DATE'))

    # print(classification_report(target, pred))
    
    output = tee.run("kl. 13.00. boka er på skolen 12.10 funker fint, men den 23.12 kl. 12.00 funker ikke. Hva med kl.14.04")
    output = tee.run("1 ukes tid")
    print(output) 
