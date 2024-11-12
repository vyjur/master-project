import configparser
from module.ere.setup import ERExtract
from module.mer.setup import MERecognition
from module.tre.setup import TRExtract
class Pipeline:
    
        def __init__(self, config_file: str):
            self.__config = configparser.ConfigParser()
            self.__config.read(config_file)
            
            self.__ere = ERExtract()
            self.__mer = MERecognition()
            self.__tre = TRExtract()
        
        def run():
            pass