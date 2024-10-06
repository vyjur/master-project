import json
import configparser
from ner.setup import Model
from preprocess.setup import Preprocess
from visualization.setup import Visualization
from pipeline.model_map import MODEL_MAP

class Pipeline:

    def __init__(self, config_file: str, train_file:str):
        
        config = configparser.ConfigParser()
        config.read(config_file)
        
        load = config['MODEL'].getboolean('load')
        
        train_parameters = {
            'train_batch_size': config.getint('train.parameters', 'train_batch_size'),
            'valid_batch_size': config.getint('train.parameters', 'valid_batch_size'),
            'epochs': config.getint('train.parameters', 'epochs'),
            'learning_rate': config.getfloat('train.parameters', 'learning_rate'),
            'shuffle': config.getboolean('train.parameters', 'shuffle'),
            'num_workers': config.getint('train.parameters', 'num_workers')
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
        
        self.__model = MODEL_MAP[config['MODEL']['name']](load, dataset, tags, train_parameters)
        self.__preprocess = Preprocess(self.__model.tokenizer)
        self.__visualization = Visualization()
        
        self.__data = None

    def run(self, data: list = []):
        preprocessed_data = self.__preprocess.run(data)
        output = self.__model.predict([val['input_ids'] for val in preprocessed_data])
        return output

    def add(self, data):
        # here calculate probability to edges?
        self.__visualization.run(self.__data)

    def predict(self, data):
        # predict new symptoms/diseases etc
        self.__visualization.run(self.__data)

if __name__ == "__main__":
    import json
    
    train_parameters = {
        'train_batch_size': 2,
        'valid_batch_size': 2,
        'epochs': 1,
        'learning_rate': 1e-04,
        'shuffle': True,
        'num_workers': 0
    }

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