import os
import configparser
from textmining.ner.setup import NERecognition

# from textmining.ere.setup import ERExtract
from textmining.tre.setup import TRExtract
from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess

# from structure.node import Node
from structure.relation import Relation
from visualization.setup import VizTool, Timeline
from pipeline.util import remove_duplicates, find_duplicates

from structure.enum import Dataset, TR_DCT
from structure.node import Node
from structure.graph import Graph

from datetime import datetime


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
            dct = datetime(2025, 1, 25, 00, 00, 00)

            output = self.__preprocess.run(doc)
            ### Text Mining ###

            ##### Perform Medical Entity Extraction
            ner_output = self.__ner.run(output)
            entities = []

            for i, _ in enumerate(output):
                result = self.__get_non_o_intervals(ner_output[i])
                start = 0
                offset = 0
                for int in result:
                    entity = self.__preprocess.decode(
                        output[i].ids[int[0] : int[1]]
                    ).strip()
                    entype = ner_output[i][int[0]].replace("B-", "")
                    found = -1
                    while found == -1 and len(
                        output[0].ids[0 : int[1] + offset]
                    ) != len(output[0].ids):
                        offset += 1
                        context = self.__preprocess.decode(
                            output[i].ids[start : int[1] + offset]
                        )
                        found = context.find(".")
                        if -1 < found < context.find(entity):
                            start = start + 1
                            offset = offset - 1
                            found = -1
                    offset = 0
                    context = (
                        context.replace("[CLS]", "")  # type: ignore
                        .replace("[SEP]", "")
                        .replace("[PAD]", "")
                    )
                    entities.append(Node(entity, entype, dct, context, None))

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
                e.dct = cat
                dcts[cat].append(e)

            ###### The candidate pairs are pairs within a group##### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
            ###### Although triple loop, this should be quicker than checking all entities
            ###### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N

            graph = Graph()

            relations = []
            for cat in dcts:
                for i, e_i in enumerate(dcts[cat]):
                    for j, e_j in enumerate(dcts[cat]):
                        if i == j:
                            continue

                        ### TODO: GET PROBABILITY
                        tre_output = self.__tre_tlink.run(e_i, e_j)

                        relation = tre_output[0]
                        prob = tre_output[1][0]

                        if relation != "O":
                            relations.append(Relation(e_i, e_j, relation, prob))

            ##### Sort relations after probability
            relations = sorted(relations, key=lambda r: r.prob, reverse=True)

            ##### Add relations one after one, make rules for consistency??!?
            for rel in relations[:]:
                graph.add_edge(rel.x.id, rel.y.id)
                if graph.is_cyclic():
                    relations.remove(rel)

            ##### Get the level ordering for the graph
            levels = graph.enumerate_levels()

            ##### Center the level ordering to the DURING group
            center = {"id": None, "lvl": 100}
            for node in levels:
                for e in entities:
                    if e.id == id:
                        e.level = levels[node]

                        if e.dct == TR_DCT.DURING and e.level < center["lvl"]:
                            center["id"] = node
                            center["lvl"] = levels[node]

            updated_levels = {
                node: level - center["id"] for node, level in levels.items()
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

        ### Visualize ###
        self.viz.create(all_info)


if __name__ == "__main__":
    pipeline = Pipeline("./src/pipeline/config.ini")
    text = "Hei på deg!"
    text2 = "Vi snakkes"
    print(pipeline.run([text, text2]))
