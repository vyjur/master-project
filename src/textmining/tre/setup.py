import configparser
import configparser
import os
from model.util import Util
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
import configparser
import random
from structure.enum import Dataset

SAVE_DIRECTORY = './src/textmining/tre'
class TRExtract:
    def __init__(self, config_file:str, manager:DatasetManager):

        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)
        
        load = self.__config['MODEL'].getboolean('load')
        
        #checkpoint = "ltg/norbert3-large"        
        checkpoint = "ltg/norbert3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        train_parameters = {
            'train_batch_size': self.__config.getint('train.parameters', 'train_batch_size'),
            'valid_batch_size': self.__config.getint('train.parameters', 'valid_batch_size'),
            'epochs': self.__config.getint('train.parameters', 'epochs'),
            'learning_rate': self.__config.getfloat('train.parameters', 'learning_rate'),
            'shuffle': self.__config.getboolean('train.parameters', 'shuffle'),
            'num_workers': self.__config.getint('train.parameters', 'num_workers'),
            'max_length': self.__config.getint('MODEL', 'max_length')
        }

        dataset_ner = manager.get(Dataset.NER)
        dataset_tre = manager.get(Dataset.TRE)
        sentences = manager.get(Dataset.SENTENCES)

        dataset = []
        tags = set()
        for k, doc in enumerate(dataset_ner):
            for i, e_i in enumerate(doc.itertuples()):
                for j, e_j in enumerate(doc.itertuples()):
                    if i == j:
                        continue
                    relations = dataset_tre[k][(dataset_tre[k]['fk_id'] == e_i[1]) & (dataset_tre[k]['id'] == e_j[1])]
                    
                    if len(relations) == 1:
                        relation = relations.iloc[0]['Temporal Relation']
                    else:
                        relation = 'O'
                        
                        # TODO: downsample majority class
                        if random.random() < 0.999:
                            continue
                    relation_pair = {
                        'i': e_i[3],
                        'context_i': sentences[k].loc[e_i[2]],
                        'context_j': sentences[k].loc[e_j[2]],
                        'j': e_j[3],
                        'relation': relation
                    }
                    dataset.append(relation_pair)
                    tags.add(relation)
        tags = list(tags)
        self.label2id, self.id2label = Util().get_tags('sequence', tags)    
        
        if load:
            self.__model = MODEL_MAP[self.__config['MODEL']['name']](load, SAVE_DIRECTORY, dataset, tags, train_parameters, self.tokenizer)
        else:
            self.__model = MODEL_MAP[self.__config['MODEL']['name']](load, SAVE_DIRECTORY, dataset, tags, train_parameters, self.tokenizer)
    
    def get_tokenizer(self):
        return self.__model.tokenizer
    
    def get_max_length(self):
        return self.__config.getint('MODEL', 'max_length')
    
    def run(self, data):
        output = self.__model.predict([val.ids for val in data])
        predictions = [self.id2label[i] for i in output]
        return predictions
        
if __name__ == "__main__":
    folder_path = "./data/annotated/"
    files = [folder_path + f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    manager = DatasetManager(files)
    
    reg = TRExtract('./src/textmining/tre/config.ini', manager)
    preprocess = Preprocess(reg.get_tokenizer(), reg.get_max_length())
    
    text = "Hei på deg!"
    
    print(reg.run(preprocess.run(text)))