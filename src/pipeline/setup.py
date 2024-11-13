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

            ### Initialize text mining modules ###
            self.__mer = MERecognition(self.__config['CONFIGS']['mer'])
            self.__ere = ERExtract(self.__config['CONFIGS']['ere'])
            self.__tre = TRExtract(self.__config['CONFIGS']['tre'])
            
            ### Initialize preprocessing module ###
            self.__preprocess = Preprocess(self.__mer.get_tokenizer(), self.__mer.get_max_length())

            ### Initialize trajectory modules ### 
            # TODO
        
            ### Initialize visualization module ###
            # TODO
            
        def __get_non_o_intervals(self, lst):
            intervals = []
            start = None
            
            for i, value in enumerate(lst):
                if value != 'O' and (start is None or not value.startswith('B')):
                    if start is None:  # Starting a new interval
                        start = i
                else:
                    if start is not None:  # Closing an existing interval
                        intervals.append((start, i))
                        start = None
            
            # If the last element is part of an interval
            if start is not None:
                intervals.append((start, len(lst) - 1))
            
            return intervals
            
        def run(self, text):
            
            ### Extract text from PDF ###
            # TODO
            
            output = self.__preprocess.run(text)
            ### Text Mining ###
            mer_output = self.__mer.run(output)
            print(mer_output)
            entities = []
            for i, doc in enumerate(output):
                result = self.__get_non_o_intervals(mer_output[i])
                start = 0
                offset = 0
                for int in result:
                    entity = self.__preprocess.decode(output[i].ids[int[0]:int[1]]).strip()
                    found = -1
                    while found == -1 and len(output[0].ids[0:int[1]+offset]) != len(output[0].ids):
                        offset += 1
                        context = self.__preprocess.decode(output[i].ids[start:int[1]+offset])
                        found = context.find('.')
                        if -1 < found < context.find(entity):
                            start = start + 1
                            offset = offset - 1
                            found = -1
                    offset = 0
                    context = context.replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '')
                    entities.append((entity, context))
            
            for i, e_i in enumerate(entities):
                for j, e_j in enumerate(entities):
                    if i == j:
                        continue
                    relation = f"{e_i[0]}: {e_i[1]} [SEP] {e_j[0]}: {e_j[1]}"
                    tokenized_relation = relation
                    tre_output = self.__tre.run(tokenized_relation)
                    ere_output = self.__ere.run(tokenized_relation)
            # self.__ere.run()

            ### Constructing trajectory ###
            
            
            ### Visualize ###
            
if __name__ == "__main__":
    pipeline = Pipeline('./src/pipeline/config.ini')
    text = "Hei på deg!"
    print(pipeline.run(text))