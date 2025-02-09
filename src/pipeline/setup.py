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
from visualization.setup import VizTool, Timeline
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

        folder_path = "./data/annotated_MTSamples/"
        files = [
            folder_path + f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]

        manager = DatasetManager(files)

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

        for i, value in enumerate(lst):
            if value != "O" and (start is None or not value.startswith("B")):
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
        # TODO: add document logic, is this necessary?

        all_info = []

        for doc in documents:
            ### Initialization
            entities = []
            graph = Graph()

            ### Preprocessing
            output = self.__preprocess.run(doc)
            
            ### Text Mining ###
            
            ##### Perform Temporal Expression Extraction
            # TODO: make some extra rules as heideltime is not extracting all type of dates that we want in Norwegian
            # Do this inside of TEE
            # Rule 1: -XX => Year
            # Rule 2: DD.MM => Same year as written
            # Rule 3: P5D => Subtract or add on the DCT
            tee_output = self.__tee.run(doc)
            
            # TODO: we need to choose what to have as DCT
            # Simple rule: First date in the page is considered as DCT
            dct = datetime(2025, 1, 25, 00, 00, 00)
            self.__tee.set_dct(dct)
            
            for te in tee_output:
                entities.append(
                    Node(te['text'], te['type'], dct, te['context'], te['value'])
                )

            ##### Perform Medical Entity Extraction
            ner_output, _ = self.__ner.run(output)
            
            all_outputs = list(itertools.chain(*output))
            
            # TODO: Config?
            WINDOW = 50
            
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

                    entype = ner_output[i][int[0]].replace("B-", "")
                    if len(entity) == 0 or entype == "O":
                        continue

                    # TODO: fix context here to window instead of sentence
                    start = sum([len(output[j]) for j in range(i)])
                    context = all_outputs[max(0, start + int[0] - WINDOW), max(len(all_outputs),start + int[1] + WINDOW)]
                    context = (
                        context.replace("[CLS]", "")  # type: ignore
                        .replace("[SEP]", "")
                        .replace("[PAD]", "")
                    )
                    curr_ent = Node(entity, entype, None, context, dct)
                    entities.append(curr_ent)
                    graph.add(curr_ent.id)

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
                if e.cat == TR_DCT.OVERLAP:
                    e.date = dct
                dcts[cat].append(e)

            ###### The candidate pairs are pairs within a group##### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
            ###### Although triple loop, this should be quicker than checking all entities
            ###### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N

            ###### TODO: add TLINK: EVENT x TIMEX
            ###### if overlap with a TIMEX. date becomes this.
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
                                if type(e_i.type) == TIMEX and type(e_j.type) != TIMEX: 
                                    if e_i.prob >= e_j.prob:
                                        e_j.date = e_i.date
                                        e_j.prob = e_j.prob
                                elif type(e_j.type) == TIMEX and type(e_i.type) != TIMEX:
                                    if e_j.prob >= e_i.prob:
                                        e_i.date = e_j.date
                                        e_i.prob = e_j.prob

                                
                            relations.append(rel)

            ##### Sort relations after probability
            relations = sorted(relations, key=lambda r: r.prob, reverse=True)

            ##### Add relations one after one, make rules for consistency??!? do we need this if we have date already?
            ##### TODO: after added TIMEX check that Relation date is consistent as well
            for rel in relations[:]:
                if rel.tr == TR_TLINK.BEFORE: 
                    graph.add_edge(rel.x.id, rel.y.id) 
                elif rel.tr == TR_TLINK.OVERLAP:
                    pass
                if graph.is_cyclic():
                    relations.remove(rel)
                    graph.remove_edge(rel.x.id, rel.y.id)

            # TODO: go through relations and for the before without any dates just put them one hour before that entity relation
            ##### Get the level ordering for the graph
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
    text = "Hei på deg!"
    text2 = "Vi snakkes"
    print(pipeline.run([text, text2]))
