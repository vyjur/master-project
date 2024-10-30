import json
import configparser
import glob
import os
import pandas as pd
from preprocess.setup import Preprocess
from visualization.setup import Visualization
from pipeline.model_map import MODEL_MAP
from transformers import AutoTokenizer
from pipeline.lexicon import Lexicon

class Pipeline:

    def __init__(self, config_file: str, train_file:str, align:bool = True):
        
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)
        
        load = self.__config['MODEL'].getboolean('load')
        
        train_parameters = {
            'train_batch_size': self.__config.getint('train.parameters', 'train_batch_size'),
            'valid_batch_size': self.__config.getint('train.parameters', 'valid_batch_size'),
            'epochs': self.__config.getint('train.parameters', 'epochs'),
            'learning_rate': self.__config.getfloat('train.parameters', 'learning_rate'),
            'shuffle': self.__config.getboolean('train.parameters', 'shuffle'),
            'num_workers': self.__config.getint('train.parameters', 'num_workers'),
            'max_length': self.__config.getint('MODEL', 'max_length'),
            'window': self.__config.getint('train.parameters', 'window')
        }
        
        checkpoint = "ltg/norbert3-large"
        # checkpoint = "ltg/norbert3-xs"
        # checkpoint = "NbAiLab/nb-bert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        if "csv" in train_file:
            dataset = pd.read_csv(train_file)
            dataset.drop(["Unnamed: 0"], axis=1, inplace=True)
            
            tags = dataset['Category'].unique()
        elif train_file == "NorSynthClinical":
            # Define the directory containing the .ann files

            if train_parameters['window'] != 0:
                directory_path = './data/NorSynthClinical'

                # Use glob to find all .ann files in the directory
                ann_files = glob.glob(os.path.join(directory_path, '*.ann'))

                tags = set()
                # Read and print the content of each .vert file
                dataset = []
                for file_path in ann_files:
                    file_dataset = []
                    with open(file_path, 'r', encoding="UTF-8") as file:        
                        for line in file:
                            data = line.split('\t')
                            if "R" in data[0]:
                                continue
                            tag = data[1].split()[0].strip()
                            if tag not in ['CONDITION', 'EVENT']:
                                tag = "O"
                            word = data[2].strip().replace('“', "").replace("”", "").split()
                            for w in word:
                                file_dataset.append((tag, w))
                            tags.add(tag)
                    dataset.append(file_dataset)
                tags = list(tags)
            else:
                file_path = './data/all_sentences.vert.entity'
                dataset = []
                tags = set()
                sentence = []
                with open(file_path, 'r', encoding='UTF-8') as file:
                    for line in file:
                        if line.startswith("#"):
                            continue
                        if line.startswith('\n'):
                            dataset.append(sentence.copy())
                            sentence = []
                            continue
                        data = line.split('\t')
                        tag = data[1].strip()
                        if tag not in ['CONDITION', 'EVENT']:
                            tag = "O"
                        word = data[0].strip().replace('“', "").replace("”", "")

                        tags.add(tag)
                        sentence.append((tag, word))
                tags = list(tags)
        else:
            with open(train_file) as f:
                d = json.load(f)

            dataset = []

            for example in d['examples']:
                
                entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
                
                dataset.append({
                    'text': example['content'],
                    'entities': entities
                })

            tags = set()

            for example in d['examples']:
                for annot in example['annotations']:
                    tags.add(annot['tag_name'])
                    
            tags = list(tags)
        
        self.__model = MODEL_MAP[self.__config['MODEL']['name']](load, dataset, tags, train_parameters, align, self.tokenizer)
        self.__preprocess = Preprocess(self.__model.tokenizer, int(self.__config['MODEL']['max_length']))
        self.label2id, self.id2label = self.__preprocess.get_tags(tags)
        self.__visualization = Visualization()
        
        self.__data = None

    def run(self, data: list = []):
        if self.__config['MODEL']['name'] != 'LLM':
            preprocessed_data = self.__preprocess.run(data)
            output = self.__model.predict([val['input_ids'] for val in preprocessed_data])
            
            predictions = [[self.id2label[int(j.cpu().numpy())] for j in i ] for i in output]
            output = predictions
            #lexi_predictions = Lexicon().predict(preprocessed_data, self.tokenizer)
            #output = Lexicon().merge(lexi_predictions, predictions)
        else:
            output = self.__model.predict([val['text'] for val in data])
        return output

    def add(self, data):
        # here calculate probability to edges?
        self.__visualization.run(self.__data)

    def predict(self, data):
        # predict new symptoms/diseases etc
        self.__visualization.run(self.__data)

if __name__ == "__main__":
    import json
    
    with open('./data/Corona2.json') as f:
        d = json.load(f)

    dataset_sample = []

    for example in d['examples']:
        
        entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
        
        dataset_sample.append({
            'text': example['content'],
            'entities': entities
        })
    
    pipeline = Pipeline('./src/pipeline/config.ini', 'NorSynthClinical', align=False)
    
    pred1 = pipeline.run(dataset_sample[0:2])   
    print(pred1)
    print(len(pred1[0]))
    
    pred2 = pipeline.run([{'text': 'Pasienten har som ledd i familiescreening fått påvist mutasjon i MYH7 som er årsak til hypertrofisk kardiomyopati.'}])
    print(pred2)