import os
import copy
import math
import pandas as pd
import configparser
from datetime import datetime

from textmining.mer.setup import MERecognition

from textmining.tre.setup import TRExtract
from textmining.tee.setup import TEExtract

from preprocess.dataset import DatasetManager
from preprocess.setup import Preprocess

from structure.relation import Relation
from visualization.setup import Timeline

from structure.enum import Dataset, DocTimeRel, TLINK, TIMEX, ME
from structure.node import Node
from structure.graph import Graph


class Pipeline:
    def __init__(self, config_file: str):
        ### Initialize configuration file ###
        self.__config = configparser.ConfigParser()
        self.__config.read(config_file)

        ### Initialize preprocessing module ###

        load = self.__config.getboolean("GENERAL", "load")

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

            manager = DatasetManager(
                entity_files,
                relation_files,
                window_size=self.__config.getint("PARAMETERS", "context"),
            )

        else:
            manager = None

        ### Initialize text mining modules ###

        print("### Initializing NER ###")
        self.__ner = MERecognition(self.__config["CONFIGS"]["ner"], manager)

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
            self.__ner.get_tokenizer(),
            self.__ner.get_max_length(),
            self.__ner.get_stride(),
            self.__ner.get_util(),
        )

        ### Initialize visualization module ###
        self.viz = Timeline()

    def run(self, documents, save_path="./", step=None, dct_cp=True):
        all_info = []
        for doc in documents:
            ### Initialization

            graph = Graph()

            ### Text Mining ###

            ##### Perform Temporal Expression Extraction

            dcts, sectimes = self.__tee.extract_sectime(doc)

            print("SECTIMES", sectimes)
            default = datetime.today().strftime("%Y-%m-%d")

            if len(sectimes) == 0:
                if len(dcts) > 0:
                    curr_date = dcts[0]["value"]
                else:
                    curr_date = default
                sectimes = [{"index": 0, "value": curr_date}]

            # For each DCT in document, define the text section corresponding to the given DCT
            for i, dct in enumerate(sectimes):
                if default != dct["value"]:
                    self.__tee.set_dct(self.__normalize_date(dct["value"]))

                print("SECTIME:", dct["value"])
                entities = []
                start = sectimes[i]["index"]
                if i + 1 >= len(sectimes):
                    stop = len(doc)
                else:
                    stop = sectimes[i + 1]["index"]
                sec_text = doc[start:stop]
                tee_output = self.__tee.run(sec_text)
                sec_text = self.__tee.pre_rules(sec_text)

                tee_output["dct"] = dct["value"]

                if step != "MER":
                    print("--- TEE:")
                    for _, te in tee_output.iterrows():
                        print(te["text"], te["value"])
                        if te["value"] == dct["value"]:
                            continue
                        entities.append(
                            Node(
                                te["text"],
                                te["type"],
                                te["dct"],
                                te["context"],
                                self.__normalize_date(te["value"]),
                            )
                        )

                ### Preprocessing based on section text
                output = self.__preprocess.run(sec_text)
                print("PAGE:", sec_text)

                ##### Perform Medical Entity Extraction
                ner_output = self.__ner.run(output)

                all_outputs = []
                for out in output:
                    all_outputs.append(
                        self.__preprocess.decode(out.ids)
                        .replace("[CLS]", "")
                        .replace("[PAD]", "")
                        .replace("[SEP]", "")
                        .strip()
                    )

                all_outputs = " ".join(all_outputs)

                WINDOW = self.__config.getint("PARAMETERS", "context")

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

                        entype = self.__ner.get_util().remove_schema(
                            ner_output[i][int[0]]
                        )
                        if len(entity) < 3 or entype == "O":
                            continue

                        start_context = (
                            self.__preprocess.decode(output[i].ids[: int[0]])
                            .replace("[CLS]", "")
                            .replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .strip()
                        ).split()
                        end_context = (
                            self.__preprocess.decode(output[i].ids[int[0] :])
                            .replace("[CLS]", "")
                            .replace("[PAD]", "")
                            .replace("[SEP]", "")
                            .strip()
                        ).split()

                        context = (
                            start_context[max(0, len(start_context) - WINDOW) :]
                            + end_context[: min(len(end_context), WINDOW)]
                        )
                        context = " ".join(context)
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

                #### Remove duplicates (document-level)
                # TODO: how to remove redundancy

                ###### Predicting each entities' DTR group

                relations = []

                if step != "MER":
                    try:
                        dct_date = datetime.strptime(dct["value"], "%Y-%m-%d")
                    except:
                        dct_date = None
                    default_date = datetime.today().date()

                    for e in entities:
                        if dct_date and dct_date.date() == default_date:
                            cat = DocTimeRel.BEFORE.name
                        elif isinstance(e.type, TIMEX):
                            cat = None
                            date = self.__normalize_date(e.date)
                            if isinstance(date, str):
                                date = datetime.strptime(date, "%Y-%m-%d")
                            if date is not None and dct_date is not None:
                                if date > dct_date:
                                    cat = DocTimeRel.AFTER.name
                                elif date < dct_date:
                                    cat = DocTimeRel.BEFORE.name
                                else:
                                    cat = DocTimeRel.OVERLAP.name
                        else:
                            e_copy = copy.deepcopy(e)
                            cat, _ = self.__tre_dtr.run(e_copy)
                            cat = cat.replace("/", "")

                            # Set date of entity as the same as DTR if it is overlapping with DCT
                            if (
                                cat == DocTimeRel.OVERLAP.name
                                and dct_date != default_date
                            ):
                                e.date = dct_date

                        print(f"++++++++++++ {i}")
                        print("TEXT:", e.value, "CAT;", cat)
                        if cat is not None:
                            dcts[cat].append(e)

                    ###### The candidate pairs are pairs within a group##### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N
                    ###### Although triple loop, this should be quicker than checking all entities
                    ###### O(N^2)>O(len(dcts)*(N_i^2)) where N_i < N

                    # TODO: go through all timex instead?
                    if not dct_cp:
                        for i, e_i in enumerate(entities):
                            print(f"######## {i}")
                            print("TEXT:", e_i.value)

                            print(e_i.context)
                            for j, e_j in enumerate(entities):
                                if not (
                                    isinstance(e_i.type, ME)
                                    and isinstance(e_j.type, TIMEX)
                                ):
                                    continue

                                if e_j.value not in e_i.context:
                                    continue

                                tre_output = self.__tre_tlink.run(e_i, e_j)

                                relation = tre_output[0]
                                prob = math.exp(tre_output[1][0])

                                print("RELATION:", relation, prob)
                                print(e_i, e_j)

                                if relation != "O":
                                    rel = Relation(e_i, e_j, relation, prob)
                                    if rel.tr == TLINK.OVERLAP:
                                        if isinstance(e_i.type, ME) and isinstance(
                                            e_j.type, TIMEX
                                        ):
                                            if (
                                                prob >= e_i.prob or e_i.date is None
                                            ) and e_j.date is not None:
                                                e_i.date = e_j.date
                                                e_i.prob = prob

                                    relations.append(rel)

                    else:
                        print(dcts)

                        for cat in dcts:
                            print("DCT:", cat)
                            for e_i in dcts[cat]:
                                print(e_i)
                                print(e_i.context)
                        for cat in dcts:
                            for i, e_i in enumerate(dcts[cat]):
                                for j, e_j in enumerate(dcts[cat]):
                                    # SKIP if not in same context or both are TIMEX

                                    if (
                                        i == j
                                        or e_j.value not in e_i.context
                                        or not (
                                            isinstance(e_i.type, ME)
                                            and isinstance(e_j.type, TIMEX)
                                        )
                                    ):
                                        continue

                                    print("\n------------------------")

                                    tre_output = self.__tre_tlink.run(e_i, e_j)

                                    relation = tre_output[0]
                                    prob = math.exp(tre_output[1][0])

                                    print(e_i, e_j)

                                    print("RELATION:", relation, prob)
                                    tre_output = self.__tre_tlink.run(e_j, e_i)

                                    n_relation = tre_output[0]
                                    n_prob = math.exp(tre_output[1][0])

                                    print("RELATION:", n_relation, n_prob)

                                    if n_prob > prob:
                                        prob = n_prob
                                        relation = n_relation

                                    if relation != "O":
                                        rel = Relation(e_i, e_j, relation, prob)

                                        if rel.tr == TLINK.OVERLAP:
                                            if isinstance(e_i.type, ME) and isinstance(
                                                e_j.type, TIMEX
                                            ):
                                                if (
                                                    prob > e_i.prob or e_i.date is None
                                                ) and e_j.date is not None:
                                                    e_i.date = e_j.date
                                                    e_i.prob = prob

                                            print(e_i)
                                        relations.append(rel)

                    print(len(relations))
                    ##### Sort relations after probability
                    relations = sorted(relations, key=lambda r: r.prob, reverse=True)

                    ### For TLINK BEFORE relation, we will add it X hours ahead before the parent entity if it has no date assigned to it.
                    ### This will be handled in the visualization module

                    # INFO: Take back if use TLINK.BEFORE and TLINK.AFTER
                    # for rel in relations:
                    # if rel.tr == TLINK.BEFORE:
                    # if None not in (rel.x.date, rel.y.date):
                    # x_date = self.__ensure_datetime(rel.x.date)
                    # y_date = self.__ensure_datetime(rel.y.date)

                    # if x_date < y_date:
                    # graph.add_edge(rel.x.id, rel.y.id)
                    # else:
                    # graph.add_edge(rel.x.id, rel.y.id)
                    # if graph.is_cyclic():
                    # relations.remove(rel)
                    # graph.remove_edge(rel.x.id, rel.y.id)

                    # for rel in relations:
                    # if rel.tr == TLINK.BEFORE:
                    # if rel.x.date is None and rel.y.date is not None and not isinstance(rel.x.type, TIMEX):
                    # rel.x.date = datetime.strptime(rel.y.date, "%Y-%m-%d") - timedelta(days=1)
                    # elif rel.x.date is not None and rel.y.date is None and not isinstance(rel.y.type, TIMEX):
                    # rel.y.date = datetime.strptime(rel.x.date, "%Y-%m-%d") + timedelta(days=1)

                final_entities = []
                for ent in entities:
                    if isinstance(ent.date, str):
                        ent_date = self.__normalize_date(ent.date)
                    elif ent.date is not None:
                        ent_date = ent.date.strftime("%Y-%m-%d")
                    else:
                        ent_date = ent.date
                    if (
                        isinstance(ent.type, TIMEX)
                        or ent_date is None
                        or ent_date == default_date
                    ):
                        continue

                    final_entities.append(ent)

                all_info.append(
                    {
                        "dct": dct_date,
                        "all_entities": entities,
                        "entities": final_entities,
                        "relations": relations,
                        "graph": graph,
                    }
                )

        ### Visualize: Using a timeline

        all_entities = []
        final_entities = []
        final_relations = []
        for info in all_info:
            all_entities.extend(info["all_entities"])
            final_entities.extend(info["entities"])
            final_relations.extend(info["relations"])

        print(len(all_entities), len(final_entities))

        for ent in all_entities:
            ent.context = ""

        df = pd.DataFrame([vars(ent) for ent in final_entities])
        if "date" in df.columns:
            df["date"] = df["date"].apply(self.__normalize_date)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(by="date")
        df.to_csv(save_path + "final.csv")

        df = pd.DataFrame([vars(ent) for ent in all_entities])
        if "date" in df.columns:
            df["date"] = df["date"].apply(self.__normalize_date)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(by="date")
        df.to_csv(save_path + "all.csv")

        df = pd.DataFrame([vars(ent) for ent in final_relations])
        df.to_csv(save_path + "relations.csv")

        self.viz.create(all_info, save_path)

    def __ensure_datetime(self, value):
        if isinstance(value, datetime):
            return value
        return datetime.strptime(value, "%Y-%m-%d")

    def __normalize_date(self, date_str):
        if isinstance(date_str, (datetime, pd.Timestamp)):
            return date_str  # Return the date if it's already a datetime object
        if date_str is None:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if fmt == "%Y":
                    return dt.strftime("%Y-01-01")
                elif fmt == "%Y-%m":
                    return dt.strftime("%Y-%m-01")
                else:
                    return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None


if __name__ == "__main__":
    pipeline = Pipeline("./src/pipeline/config.ini")
    text = "Pasienten var innlagt 21. juni for prostata cancer men ble utskrevet dagen etter på grunn av mangel på personell. Han ble sagt til å komme tilbake uken etter for å få operasjon."
    text2 = "Medikasjonen blir det samme de neste to ukene. Men paracetamol blir økt til 2 ganger daglig og det ble sendt en henvisning til røntgen test snarest."
    pipeline.run([text, text2])
