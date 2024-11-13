import configparser
from transformers import AutoTokenizer
from textmining.ere.setup import ERExtract
from textmining.mer.setup import MERecognition
from textmining.tre.setup import TRExtract
from preprocess.setup import Preprocess
class Pipeline:
    
        def __init__(self, config_file: str):
            
            ### Initialize configuration file ###
            self.__config = configparser.ConfigParser()
            self.__config.read(config_file)
            
            ### Initialize preprocessing module ###
            checkpoint = "ltg/norbert3-small"
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

            self.__preprocess = Preprocess(self.__config['CONFIGS']['preprocess'])
            
            ### Initialize text mining modules ###
            self.__mer = MERecognition(self.__config['CONFIGS']['mer'])
            self.__ere = ERExtract(self.__config['CONFIGS']['ere'])
            self.__tre = TRExtract(self.__config['CONFIGS']['tre'])
            
            self.__preprocess = Preprocess(self.__mer.get_tokenizer(), self.__mer.get_max_length())

            ### Initialize trajectory modules ### 
            # TODO
        
            ### Initialize visualization module ###
            # TODO
            
        def run(self, text):
            
            ### Extract text from PDF ###
            # TODO
            
            self.__preprocess.run()
            
            ### Text Mining ###
            self.__mer.run()
            self.__ere.run()
            self.__tre.run()

            ### Constructing trajectory ###
            
            
            ### Visualize ###