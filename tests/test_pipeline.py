import torch
import unittest
import configparser
from pipeline.setup import Pipeline
from model.map import MODEL_MAP

class TestPipeline(unittest.TestCase):

    @unittest.skip(reason="Temporarily disabling it as it causes some issue when all models run together at once.")
    def test_pipeline(self):
        
        config_file = './tests/test_pipeline.ini'
        
        torch.cuda.empty_cache()

        config = configparser.ConfigParser()
        config.read(config_file)
        
        pipeline = Pipeline(config_file)
        
        pred1 = pipeline.run(["Hei på deg!"])  
        print(pred1)

if __name__ == '__main__':
    unittest.main()