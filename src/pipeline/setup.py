import os
import configparser
from preprocess.util import majority_element
from transformers import AutoTokenizer
from textmining.ere.setup import ERExtract
from textmining.ner.setup import NERecognition
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess
from structure.node import Node
from structure.relation import Relation
from visualization.setup import VizTool
from pipeline.util import remove_duplicates, find_duplicates
class Pipeline:
    
    def __init__(self, config_file: str):
        
        ### Initialize configuration file ###
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)
        
        ### Initialize preprocessing module ###
        # TODO: config?
        checkpoint = "ltg/norbert3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        folder_path = "./data/annotated/"
        files = [folder_path + f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        manager = DatasetManager(files)

        ### Initialize text mining modules ###
        self.__ner = NERecognition(self.__config['CONFIGS']['ner'], manager)
        self.__ere = ERExtract(self.__config['CONFIGS']['ere'], manager)
        self.__tre = TRExtract(self.__config['CONFIGS']['tre'], manager)
        
        ### Initialize preprocessing module ###
        self.__preprocess = Preprocess(self.__ner.get_tokenizer(), self.__ner.get_max_length())

        ### Initialize trajectory modules ### 
        # TODO
        
        ### Initialize visualization module ###
        self.viz = VizTool()
        
        
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
        
    def run(self, documents):
        
        ### Extract text from PDF ###
        # TODO: add document logic
        
        rel_entities = []
        
        for doc in documents:
            output = self.__preprocess.run(doc)
            ### Text Mining ###
            ner_output = self.__ner.run(output)
            entities = []
            for i, __ in enumerate(output):
                result = self.__get_non_o_intervals(ner_output[i])
                start = 0
                offset = 0
                for int in result:
                    entity = self.__preprocess.decode(output[i].ids[int[0]:int[1]]).strip()
                    entype = ner_output[i][int[0]].replace('B-', '')
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
                    entities.append((entity, context, entype))
            
            ### TODO: choose candidate pairs
            for i, e_i in enumerate(entities):      
                for j, e_j in enumerate(entities):
                    if i == j:
                        continue
                    relation = f"{e_i.value}: {e_i.context} [SEP] {e_j.value}: {e_j.context}"
                    tokenized_relation = relation
                    tre_output = majority_element(self.__tre.run(tokenized_relation))
                    ere_output = majority_element(self.__ere.run(tokenized_relation))
                    if tre_output != 'O' and ere_output != 'O':
                        e_i.relations.append(Relation(e_i, e_j, tre_output, ere_output))

            ### Remove local duplicates
            duplicates = find_duplicates(entities)
            entities = remove_duplicates(entities, duplicates)
            
            rel_entities.append(entities)
            
        ### Constructing trajectory ###            
        ### Add edges between duplicates across documents
        for i in range(len(rel_entities)-1):
            check_entities = []
            if i != 0:
                
                check_entities = rel_entities[i-1]
                check_entities = check_entities + rel_entities[i]
                duplicates = find_duplicates(check_entities, False)
                rel_entities[i-1] = remove_duplicates(rel_entities[i-1], [j for j in duplicates if j < len(rel_entities[i-1])])
                rel_entities[i] = remove_duplicates(rel_entities[i], [j - len(rel_entities[i-1]) for j in duplicates if j >= len(rel_entities[i-1])])
            
            check_entities = rel_entities[i] + rel_entities[i+1]
            duplicates = find_duplicates(check_entities, False)
            rel_entities[i] = remove_duplicates(rel_entities[i], [j for j in duplicates if j < len(rel_entities[i])])
            rel_entities[i+1] = remove_duplicates(rel_entities[i+1], [j - len(rel_entities[i]) for j in duplicates if j >= len(rel_entities[i])])
        
        all_entities = []
        for doc in rel_entities:
            all_entities = all_entities + doc
            
        ### Visualize ###
        self.viz.create(all_entities)
        
if __name__ == "__main__":
    pipeline = Pipeline('./src/pipeline/config.ini')
    text = "Hei på deg!"
    print(pipeline.run([text]))