import json
import configparser
from ner.setup import Model
from preprocess.setup import Preprocess
from visualization.setup import Visualization
from pipeline.model_map import MODEL_MAP

class Pipeline:

    def __init__(self, config_file: str, train_file:str):
        
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)
        
        load = self.__config['MODEL'].getboolean('load')
        
        train_parameters = {
            'train_batch_size': self.__config.getint('train.parameters', 'train_batch_size'),
            'valid_batch_size': self.__config.getint('train.parameters', 'valid_batch_size'),
            'epochs': self.__config.getint('train.parameters', 'epochs'),
            'learning_rate': self.__config.getfloat('train.parameters', 'learning_rate'),
            'shuffle': self.__config.getboolean('train.parameters', 'shuffle'),
            'num_workers': self.__config.getint('train.parameters', 'num_workers')
        }
        
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
        
        self.__model = MODEL_MAP[self.__config['MODEL']['name']](load, dataset, tags, train_parameters)
        self.__preprocess = Preprocess(self.__model.tokenizer)
        self.label2id, self.id2label = self.__preprocess.get_tags(tags)
        self.__visualization = Visualization()
        
        self.__data = None

    def run(self, data: list = []):
        if self.__config['MODEL']['name'] != 'LLM':
            preprocessed_data = self.__preprocess.run(data)
            output = self.__model.predict([val['input_ids'] for val in preprocessed_data])
            return [[self.id2label[int(j.numpy())] for j in i ] for i in output]
        else:
            output = self.__model.predict(data)
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
    
    pipeline = Pipeline('./src/pipeline/config.ini', './data/Corona2.json')
    
    pred1 = pipeline.run(dataset_sample[0:2])    
    print(pred1)