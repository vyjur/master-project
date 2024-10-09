import unittest
import json
import configparser
from pipeline.setup import Pipeline, MODEL_MAP

class TestPipeline(unittest.TestCase):

    def test_ner_models(self):
        with open('./data/Corona2.json') as f:
            d = json.load(f)
            
        models = MODEL_MAP.keys()
        config_file = './tests/test_pipeline.ini'
        
        for model in models:
            
            config = configparser.ConfigParser()
            config.read(config_file)
            
            config['MODEL']['name'] = model
            with open(config_file, 'w') as f:
                config.write(f)
            
            dataset_sample = []

            for example in d['examples']:
                
                entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
                
                dataset_sample.append({
                    'text': example['content'],
                    'entities': entities
                })
            
            pipeline = Pipeline(config_file, './data/Corona2.json')
            
            pred1 = pipeline.run(dataset_sample[0:2])  

            for pred in pred1:
                for val in pred:
                    if val in pipeline.label2id.keys():
                        pass
                    else:
                        raise TypeError(val, 'Output is invalid.')

if __name__ == '__main__':
    unittest.main()