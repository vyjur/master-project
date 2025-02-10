from textmining.tee.heideltime.python_heideltime import Heideltime
from textmining.tee.rules import *
import xml.etree.ElementTree as ET
import pandas as pd


class TEExtract:
    
    def __init__(self, rules:bool=True):
        self.__heideltime = Heideltime()
        self.__heideltime.set_document_type('NEWS')
        self.__heideltime.set_language('auto-norwegian')
        self.__rules = rules
        
    def set_dct(self, dct):
        self.__heideltime.set_document_time(dct)
        
    def __pre_rules(self, text):
        text = convert_text(text)
        
        text = convert_slash_date(text)
        text = convert_date_format(text)
                
        # Rule 2: -78 => 1.1.1978
        text = convert_negative_years(text)
        
        # Rule 3: 2020 => 1.1.2020
        text = convert_full_year(text)
        
        if self.__heideltime.document_time is not None:
            # Rule 4: 25.12 => 25.12.YYYY where YYYY is the same year as DCT if 25.12.YYYY < DCT. Else, the year before that.
            dct = datetime.strptime(self.__heideltime.document_time, "%Y-%m-%d")
            
            try:
                text = convert_partial_dates(text, dct)
            except:
                pass
        return text
            
    def __post_rules(self, text, ttype, value):
        if self.__heideltime.document_time is not None:
            if ttype == "DURATION":
                
                # TODO: WHAT TO DO HERE
                pass      
            
        return value 
        
    def __run(self, text):
        
        if self.__rules:
            text = self.__pre_rules(text)
        
        result = self.__heideltime.parse(text)
        
        root = ET.fromstring(result)
        timex_elements = root.findall('.//TIMEX3')
        
        data = []
        for timex in timex_elements:
            tid = timex.get('tid')
            ttype = timex.get('type')
            value = timex.get('value')
            text = timex.text
            
            if self.__rules: 
                value = self.__post_rules(text, ttype, value) 
            
            # Get the full text of the document
            full_text = "".join(root.itertext())
            
            # Get the 50-character window around this TIMEX3 element
            char_window = self.__get_char_window(full_text, text, window_size=50)
            
            data.append( {
                'id': tid,
                'type': ttype,
                'value': value,
                'text': text,
                'context': char_window
            })
        return pd.DataFrame(data)
            
    def run(self, data):
        return [self.__run(text) for text in data]
        
    def __get_char_window(self, full_text, timex_text, window_size=50):
        # Find the index of the TIMEX3 text in the full text
        start_index = full_text.find(timex_text)
        end_index = start_index + len(timex_text)
        
        # Get the character window before and after the TIMEX3 text
        before_window = full_text[max(0, start_index - window_size):start_index]  # 50 characters before TIMEX3
        after_window = full_text[end_index:end_index + window_size]  # 50 characters after TIMEX3
        
        # Combine the before and after windows with the TIMEX3 text itself in the middle
        window = before_window + timex_text + after_window
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
     
    manager = DatasetManager(entity_files, relation_files, False)
    
    dataset = manager.get(Dataset.TEE)
    
    print(dataset)

    target = []
    pred = []
    for i, data in dataset.iterrows():
        output = tee.run([data['Text']])[0]
        if not output.empty:
            pred.append(output["type"].values[0])
            target.append(data['TIMEX'].replace('DCT', 'DATE'))
        else:
            pred.append(None)
            target.append(data['TIMEX'].replace('DCT', 'DATE'))

    print(classification_report(target, pred))
    
    output = tee.run(["21/12/2021 2/1/20"])
    print(output[0]) 