from textmining.tee.heideltime.python_heideltime.python_heideltime import Heideltime
from textmining.tee.rules import *
import xml.etree.ElementTree as ET
from preprocess.dataset import DatasetManager
import configparser
from structure.enum import Task, Dataset, TIMEX
from model.util import Util
import pandas as pd
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from model.map import MODEL_MAP


class TEExtract:
    
    def __init__(self, config_file:str, manager: DatasetManager, save_directory: str = "./src/textmining/ner/model", rules:bool=True):
        self.__heideltime = Heideltime()
        self.__heideltime.set_document_type('NEWS')
        self.__heideltime.set_language('auto-norwegian')
        self.__rules = rules
        
        self.__config = configparser.ConfigParser(allow_no_value=True)
        self.__config.read(config_file)

        load = self.__config["MODEL"].getboolean("load")
        print("LOAD", load)
        
        dataset = []
        tags = [TIMEX.DATE.name, TIMEX.DCT.name]   

        if not load:
            dataset = manager.get(Dataset.TEE)
            dataset = dataset[dataset['TIMEX'].isin(["DATE", "DCT"])]
            for _, row in dataset.iterrows():
                dataset.append(
                    {
                        "sentence": row['Context']
                        .replace(row['Text'], f"<TAG>{row['Text']}</TAG>"),
                        "relation": row['TIMEX'],
                    }
                )
                tags.add(row['DCT'])
        self.label2id, self.id2label = Util().get_tags(Task.SEQUENCE, tags)
        
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

        self.__model = MODEL_MAP[self.__config["MODEL"]["name"]](
            load,
            save_directory,
            dataset,
            tags,
            parameters=train_parameters,
            tokenizer=self.tokenizer,
            project_name=self.__config["GENERAL"]["name"],
            pretrain=self.__config["pretrain"]["name"],
        )
        
        self.preprocess = Preprocess(self.get_tokenizer(), self.get_max_length())
        
    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("MODEL", "max_length")

    def __model_run(self, data):
        output, _ = self.__model.predict([val.ids for val in data])
        predictions = [[self.id2label[int(j.cpu().numpy())] for j in i] for i in output]
        return predictions

    def get_model(self):
        return self.__model
        
    def set_dct(self, dct):
        self.__heideltime.set_document_time(dct)
        
    def __pre_rules(self, text):
        text = convert_text(text)
        
        text = convert_slash_date(text)
        text = convert_date_format(text)
        text = convert_negative_years(text)
        text = convert_full_year(text)
        
        if self.__heideltime.document_time is not None:
            # Rule 4: 25.12 => 25.12.YYYY where YYYY is the same year as DCT if 25.12.YYYY < DCT. Else, the year before that.
            dct = datetime.strptime(self.__heideltime.document_time, "%Y-%m-%d")
            
            try:
                text = convert_partial_dates(text, dct)
            except:
                pass
        return text
            
    def __post_rules(self, full_text, text, ttype, value):
        dct = datetime.strptime(self.__heideltime.document_time, "%Y-%m-%d")
        check_context = self.__get_window(full_text, text, window_size=3)

        if self.__heideltime.document_time is not None:
            if ttype == "DURATION":
                return convert_duration(check_context, value, dct) 
        return value 
        
    def __run(self, text):
        if self.__rules:
            text = self.__pre_rules(text)
        
        result = self.__heideltime.parse(text)
        
        root = ET.fromstring(result)
        full_text = " ".join(root.itertext())

        timex_elements = root.findall('.//TIMEX3')
        
        data = []
        for timex in timex_elements:
            tid = timex.get('tid')
            ttype = timex.get('type')
            value = timex.get('value')
            text = timex.text
            
            # Get the full text of the document
            
            # Get the 50-token window around this TIMEX3 element
            context = self.__get_window(full_text, text, window_size=50)
            
            if self.__rules: 
                value = self.__post_rules(full_text, text, ttype, value)
            
            data.append( {
                'id': tid,
                'type': ttype,
                'value': value,
                'text': text,
                'context': context
            })

        return pd.DataFrame(data)
            
    def run(self, data):
        # Initial output: Extracting all TIMEX expressions
        init_output = self.__run(data)
       
        # Only DATE expressions are candidate for DCT 
        dct_candidates = init_output[init_output['type'] == 'DATE']
        
        # For each candidate classify if it is really a DCT or not
        dcts = []
        sections = []

        for i, row in dct_candidates.iterrows():
            text = row['context'].replace(row['text'], f"<TAG>{row['text']}</TAG>")
            dct_output = self.__model_run(text)
            
            if dct_output == "DCT":
                dcts.append(row)
                context_start = data.index(dct['context'])
                dct_start = dct['context'].index(dct['text'])
                start = context_start + dct_start
                sections.append(start)
            
        processed_output = []
       
        # For each DCT in document, define the text section corresponding to the given DCT 
        for i, dct in enumerate(dcts):
            if i + 1 > len(sections):
                end = len(data)
            else:
                end = i + 1
            sec_text = data[sections[i], end] 
            sec_output = self.__run(sec_text)
            sec_output['dct'] = dct['value']
            processed_output.append(sec_output)
            
        return pd.concat(processed_output, axis=1).reset_index()
        
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