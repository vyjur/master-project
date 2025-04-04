import os
import pandas as pd
import configparser
from datetime import datetime, timedelta

from textmining.ner.setup import NERecognition

from textmining.tre.setup import TRExtract
from textmining.tee.setup import TEExtract

from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess

from structure.relation import Relation
from visualization.setup import Timeline
from pipeline.util import remove_duplicates, find_duplicates

from structure.enum import Dataset, DocTimeRel, TLINK, TIMEX
from structure.node import Node
from structure.graph import Graph


class Pipeline:
    def __init__(self, config_file: str):
        ### Initialize configuration file ###
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)

        ### Initialize preprocessing module ###
        
        load = self.__config.getboolean('GENERAL', 'load')
        
        if not load:
        
            folder_path = "./data/helsearkiv/annotated/entity/"

            entity_files = [
                folder_path + f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]

            folder_path = "./data/helsearkiv/annotated/relation/"

            relation_files = [
                folder_path + f
                for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f))
            ]

            manager = DatasetManager(entity_files, relation_files, window_size=self.__config.getint("PARAMETERS", 'context'))

        else:
            manager = None
            
        ### Initialize text mining modules ###

        print("### Initializing NER ###")
        self.__ner = NERecognition(self.__config["CONFIGS"]["ner"], manager)
        
        print("### Initializing TEE ###")
        self.__tee = TEExtract(self.__config["CONFIGS"]["tee"], manager)

        print("### Initializing DTR ###")
        self.__tre_dtr = TRExtract(
            self.__config["CONFIGS"]["tre_dtr"], manager, Dataset.DTR
        )

        print("### Initializing TLINK ###")
        self.__tre_tlink = TRExtract(
            self.__config["CONFIGS"]["tre_tlink"], manager, Dataset.TLINK
        )

        ### Initialize preprocessing module ###
        self.__preprocess = Preprocess(
            self.__ner.get_tokenizer(), self.__ner.get_max_length(), self.__ner.get_stride(), self.__ner.get_util()
        )

        ### Initialize visualization module ###
        self.viz = Timeline()

    

    def run(self, documents, save_path='./', step=None):
        all_info = []
        for doc in documents:
            entities = []
            
            ### Initialization
            
            graph = Graph()
            
            ### Text Mining ###
            
            ##### Perform Temporal Expression Extraction
        
            dcts, sectimes = self.__tee.extract_sectime(doc)
            
            print("SECTIMES", sectimes)
            
            if len(sectimes) == 0:
                if len(dcts) > 0:
                    default = dcts[0]['value']
                else:
                    default = datetime.today().strftime("%Y-%m-%d")
                sectimes = [{
                    "index": 0,
                    "value": default
                }]
       
            # For each DCT in document, define the text section corresponding to the given DCT 
            for i, dct in enumerate(sectimes):
                start = sectimes[i]["index"]
                if i + 1 >= len(sectimes):
                    stop = len(doc)
                else:
                    stop = sectimes[i+1]["index"]
                sec_text = doc[start:stop] 
                tee_output = self.__tee.run(sec_text)
                tee_output['dct'] = dct['value']
                          
                if step != "MER":  
                    for _, te in tee_output.iterrows():
                        entities.append(
                            Node(te['text'], te['type'], te['dct'], te['context'], te['value'])
                        )
                    
                ### Preprocessing based on section text
                output = self.__preprocess.run(sec_text)

                ##### Perform Medical Entity Extraction
                ner_output = self.__ner.run(output)
                
                all_outputs = []
                for out in output:
                    all_outputs.append(self.__preprocess.decode(out.ids)
                            .replace("[CLS]", "")
                            .replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .strip())
                
                all_outputs = " ".join(all_outputs)
                
                WINDOW = self.__config.getint('PARAMETERS', 'context')
                
                for i, _ in enumerate(output):
                    
                    result = self.__ner.get_non_o_intervals(ner_output[i])
                    start = 0
                    for int in result:
                        entity = (
                            self.__preprocess.decode(output[i].ids[int[0] : int[1]])
                            .replace("[CLS]", "")
                            .replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .strip()
                        )

                        entype = self.__ner.get_util().remove_schema(ner_output[i][int[0]])
                        if len(entity) == 0 or entype == "O":
                            continue

                        # Token window based on tokenization output
                        start = sum([len(output[j]) for j in range(i)])
                        # TODO: context here is wrong, need to get token context instead
                        start_context = all_outputs[:start].split()
                        end_context = all_outputs[start:].split()
                        context = start_context[max(0, len(start_context) - WINDOW):] + end_context[:min(len(end_context), WINDOW)]
                        context = " ".join(context)
                        print("##################")
                        print("CONTEXT:", len(context))
                        print(context)
                        context = (
                            context.replace("[CLS]", "")  # type: ignore
                            .replace("[SEP]", "")
                            .replace("[PAD]", "")
                        )
                        
                        curr_ent = Node(entity, entype, None, context, None)
                        entities.append(curr_ent)
                        graph.add_node(curr_ent.id)

                ### Temporal Relation Extraction

                #### DocTimeRel Extraction
                dcts = {}

                ###### Initialize groups for selecting candidate pairs
                for cat in DocTimeRel:
                    dcts[cat.name] = []

                #### Remove local duplicates (document-level)
                # TODO: how can remove redundancy
                # duplicates = find_duplicates(entities)
                # entities = remove_duplicates(entities, duplicates)

                ###### Predicting each entities' DTR group
                
                relations = []
                
                if step != "MER":
                    
                    dct_date = datetime.strptime(dct['value'], "%Y-%m-%d")
                    default_date = datetime.today().strftime("%Y-%m-%d")

                    for e in entities:
                        if dct_date == default_date:
                            cat = DocTimeRel.BEFORE.name
                        elif isinstance(e.type, TIMEX):
                            date = datetime.strptime(e.date, "%Y-%m-%d")
                            if date == dct_date:
                                cat = DocTimeRel.OVERLAP.name
                            elif date < dct_date:
                                cat = DocTimeRel.BEFORE.name
                            else:
                                cat = DocTimeRel.AFTER.name
                        else:
                            cat, _ = self.__tre_dtr.run(e)
                            cat = cat.replace("/", "")
                        
                            # Set date of entity as the same as DTR if it is overlapping with DCT
                            if cat == DocTimeRel.OVERLAP:
                                e.date = e.dct
                        dcts[cat].append(e)

                    ###### The candidate pairs are pairs within a group##### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
                    ###### Although triple loop, this should be quicker than checking all entities
                    ###### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
                    

                    for cat in dcts:
                        for i, e_i in enumerate(dcts[cat]):
                            for j, e_j in enumerate(dcts[cat]):
                                
                                # SKIP if not in same context or both are TIMEX
                                if i == j or e_j.value not in e_i.context or (isinstance(e_i.type, TIMEX) and isinstance(e_j.type, TIMEX)):
                                    continue

                                tre_output = self.__tre_tlink.run(e_i, e_j)

                                relation = tre_output[0]
                                prob = tre_output[1][0]

                                if relation != "O":
                                    rel = Relation(e_i, e_j, relation, prob)
                                    if rel.tr == TLINK.OVERLAP:
                                        if isinstance(e_i.type, TIMEX) and not isinstance(e_j.type, TIMEX):
                                            if e_i.prob >= e_j.prob or e_j.date is None:
                                                e_j.date = e_i.date
                                                e_j.prob = e_i.prob 
                                        elif isinstance(e_j.type, TIMEX) and not isinstance(e_i.type, TIMEX):
                                            if e_j.prob >= e_i.prob or e_i.date is None:
                                                e_i.date = e_j.date
                                                e_i.prob = e_j.prob
                                                    
                                    relations.append(rel)

                    ##### Sort relations after probability
                    relations = sorted(relations, key=lambda r: r.prob, reverse=True)
                                
                    ### For TLINK BEFORE relation, we will add it X hours ahead before the parent entity if it has no date assigned to it.
                    ### This will be handled in the visualization module
                    for rel in relations:
                        if rel.tr == TLINK.BEFORE: 
                            if None not in (rel.x.date, rel.y.date):
                                if rel.x.date < rel.y.date:
                                    graph.add_edge(rel.x.id, rel.y.id)
                            else:
                                graph.add_edge(rel.x.id, rel.y.id)
                        if graph.is_cyclic():
                            relations.remove(rel)
                            graph.remove_edge(rel.x.id, rel.y.id)

                    for rel in relations:
                        if rel.tr == TLINK.BEFORE:
                            if rel.x.date is None and rel.y.date is not None and not isinstance(rel.x.type, TIMEX):
                                rel.x.date = datetime.strptime(rel.y.date, "%Y-%m-%d") - timedelta(days=1)
                            elif rel.x.date is not None and rel.y.date is None and not isinstance(rel.y.type, TIMEX):
                                rel.y.date = datetime.strptime(rel.x.date, "%Y-%m-%d") + timedelta(days=1)

                all_info.append(
                    {
                        "entities": entities,
                        "relations": relations,
                        "graph": graph,
                    }
                )
            
        ### Visualize: Using a timeline
        
        all_entities = []
        for info in all_info:
            all_entities.extend(info['entities'])
            
        print(len(all_entities))
        
        for ent in all_entities:
            ent.context = ""
             
        df = pd.DataFrame([vars(ent) for ent in all_entities])
        df.to_csv(save_path + "output.csv")

        self.viz.create(all_info, save_path)


if __name__ == "__main__":
    pipeline = Pipeline("./src/pipeline/config.ini")
    text = "Pasienten var innlagt 21. juni for prostata cancer men ble utskrevet dagen etter på grunn av mangel på personell. Han ble sagt til å komme tilbake uken etter for å få operasjon."
    text2 = "Medikasjonen blir det samme de neste to ukene. Men paracetamol blir økt til 2 ganger daglig og det ble sendt en henvisning til røntgen test snarest."
    pipeline.run([text, text2])
