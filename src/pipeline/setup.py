import os
import configparser
from textmining.ner.setup import NERecognition

# from textmining.ere.setup import ERExtract
from textmining.tre.setup import TRExtract
from textmining.tee.setup import TEExtract

from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess

# from structure.node import Node
from structure.relation import Relation
from visualization.setup import Timeline
from pipeline.util import remove_duplicates, find_duplicates

from structure.enum import Dataset, TR_DCT, TR_TLINK, TIMEX
from structure.node import Node
from structure.graph import Graph

from datetime import datetime
import itertools


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
        self.__tee = TEExtract()

        print("### Initializing DCT ###")
        self.__tre_dct = TRExtract(
            self.__config["CONFIGS"]["tre_dct"], manager, Dataset.TRE_DCT
        )

        print("### Initializing TLINK ###")
        self.__tre_tlink = TRExtract(
            self.__config["CONFIGS"]["tre_tlink"], manager, Dataset.TRE_DCT
        )

        ### Initialize preprocessing module ###
        self.__preprocess = Preprocess(
            self.__ner.get_tokenizer(), self.__ner.get_max_length()
        )

        ### Initialize trajectory modules ###
        # TODO: should we move some of the tasks in pipeline method into a module?

        ### Initialize visualization module ###
        self.viz = Timeline()

    def __get_non_o_intervals(self, lst):
        intervals = []
        start = None
        
        prev_value = "O"

        for i, value in enumerate(lst):
            cat_value = value.replace('B-', '').replace('I-', '')
            if value != "O":
                    
                if (value.startswith("B-") or cat_value != prev_value) and start is not None:
                    intervals.append((start, i))
                    start = None
                    
                if value.startswith("B-") or (value.startswith("I-") and start is None):
                    start = i
            else:
                if start is not None:
                    intervals.append((start, i))
                    start = None
                    
            prev_value = cat_value
                    
        # If the last element is part of an interval
        if start is not None:
            intervals.append((start, len(lst)))
            
        return intervals

    def run(self, documents):
        ### Extract text from PDF ###
        # TODO: add document logic, is this necessary?

        all_info = []
        entities = []
        for doc in documents:
            ### Initialization
            
            graph = Graph()

            ### Preprocessing
            output = self.__preprocess.run(doc)
            
            ### Text Mining ###
            
            ##### Perform Temporal Expression Extraction
            
            # TODO: we need to choose what to have as DCT
            # Simple rule: First date in the page is considered as DCT
            dct = datetime(2025, 1, 25, 00, 00, 00).strftime("%Y-%m-%d")
            self.__tee.set_dct(dct)
            
            tee_output = self.__tee.run(doc)
            
            
            for _, te in tee_output.iterrows():
                entities.append(
                    Node(te['text'], te['type'], dct, te['context'], te['value'])
                )

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
                
                result = self.__get_non_o_intervals(ner_output[i])
                start = 0
                for int in result:
                    entity = (
                        self.__preprocess.decode(output[i].ids[int[0] : int[1]])
                        .replace("[CLS]", "")
                        .replace("[PAD]", "")
                        .replace("[SEP]", "")
                        .strip()
                    )

                    entype = ner_output[i][int[0]].replace("B-", "").replace("I-", "")
                    if len(entity) == 0 or entype == "O":
                        continue

                    # Token window based on tokenization output
                    start = sum([len(output[j]) for j in range(i)])
                    context = all_outputs[max(0, start + int[0] - WINDOW) : min(len(all_outputs),start + int[1] + WINDOW)]
                    
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
            for cat in TR_DCT:
                dcts[cat.name] = []

            #### Remove local duplicates (document-level)
            duplicates = find_duplicates(entities)
            entities = remove_duplicates(entities, duplicates)

            ###### Predicting each entities' DCT group
            for e in entities:
                cat, _ = self.__tre_dct.run(e)
                cat = cat.replace("/", "")
                e.set_dct(cat)
                
                # Set date of entity as the same as DCT if it is overlapping with DCT
                if cat == TR_DCT.OVERLAP:
                    print("DCT overlap")
                    print(e)
                    e.date = dct
                dcts[cat].append(e)

            ###### The candidate pairs are pairs within a group##### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
            ###### Although triple loop, this should be quicker than checking all entities
            ###### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N

            relations = []
            for cat in dcts:
                for i, e_i in enumerate(dcts[cat]):
                    for j, e_j in enumerate(dcts[cat]):
                        if i == j:
                            continue

                        tre_output = self.__tre_tlink.run(e_i, e_j)

                        relation = tre_output[0]
                        prob = tre_output[1][0]

                        if relation != "O":
                            rel = Relation(e_i, e_j, relation, prob)
                            if rel.tr == TR_TLINK.OVERLAP:
                                if isinstance(e_i.type, TIMEX) and not isinstance(e_j.type, TIMEX):
                                    if e_i.prob >= e_j.prob or e_j.date is None:
                                        e_j.date = e_i.date
                                        e_j.prob = e_i.prob 
                                elif isinstance(e_j.type, TIMEX) and not isinstance(e_i.type, TIMEX):
                                    if e_j.prob >= e_i.prob or e_i.date is None:
                                        e_i.date = e_j.date
                                        e_i.prob = e_j.prob
                                            
                            print(rel)
                            relations.append(rel)

            ##### Sort relations after probability
            relations = sorted(relations, key=lambda r: r.prob, reverse=True)
                        
            ### For TLINK BEFORE relation, we will add it X hours ahead before the parent entity if it has no date assigned to it.
            ### This will be handled in the visualization module
            for rel in relations[:]:
                if rel.tr == TR_TLINK.BEFORE: 
                    if None not in (rel.x.date, rel.y.date):
                        if rel.x.date < rel.y.date:
                            graph.add_edge(rel.x.id, rel.y.id)
                    else:
                        graph.add_edge(rel.x.id, rel.y.id)
                elif rel.tr == TR_TLINK.OVERLAP:
                    # TODO: do we need to do something here?
                    pass
                if graph.is_cyclic():
                    relations.remove(rel)
                    graph.remove_edge(rel.x.id, rel.y.id)
                    
            ##### Get the level ordering for the graph
            
            ## INFORMATION: I don't think we need levels anymore, but DATETIME is our level instead
            levels = graph.enumerate_levels()

            ##### Center the level ordering to the OVERLAP group
            center = {"id": None, "lvl": 100}
            for node in levels:
                for e in entities:
                    if e.id == node:
                        e.level = levels[node]

                        if e.dct == TR_DCT.OVERLAP and e.level < center["lvl"]:
                            center["id"] = node
                            center["lvl"] = levels[node]

            updated_levels = {
                node: level - center["lvl"] for node, level in levels.items()
            }

            for node in updated_levels:
                for e in entities:
                    if e.id == id:
                        e.level = updated_levels[node]

            all_info.append(
                {
                    "dct": dct,
                    "entities": entities,
                    "relations": relations,
                    "graph": graph,
                }
            )
        ## TODO: fix so date is used instead of the level things we have been using right now maybe the one above not needed.
        ### Visualize ###
        self.viz.create(all_info)


if __name__ == "__main__":
    pipeline = Pipeline("./src/pipeline/config.ini")
    text = "Pasienten var innlagt 21. juni for prostata cancer men ble utskrevet dagen etter på grunn av mangel på personell. Han ble sagt til å komme tilbake uken etter for å få operasjon."
    text2 = "Medikasjonen blir det samme de neste to ukene. Men paracetamol blir økt til 2 ganger daglig og det ble sendt en henvisning til røntgen test snarest."
    pipeline.run([text, text2])
