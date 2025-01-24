import configparser
import os
from model.util import Util
from textmining.ner.lexicon import Lexicon
from preprocess.dataset import DatasetManager
from model.map import MODEL_MAP
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from structure.enum import Dataset, Task


class NERecognition:
    def __init__(
        self,
        config_file: str,
        manager: DatasetManager,
        save_directory: str = "./src/textmining/ner/model",
    ):
        self.__config = configparser.ConfigParser(allow_no_value=True)
        self.__config.read(config_file)

        load = self.__config["MODEL"].getboolean("load")
        raw_dataset = manager.get(Dataset.NER)

        dataset = []
        tags = set()
        for doc in raw_dataset:
            curr_doc = []
            for row in doc.itertuples(index=False):
                curr_doc.append((row[2], row[3]))  # Add (row[1], row[2]) tuple to list
                tags.add(row[3])  # Add row[2] to the set

            dataset.append(curr_doc)
        tags = list(tags)
        self.label2id, self.id2label = Util().get_tags(Task.TOKEN, tags)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.__config["pretrain"]["name"]
        )

        train_parameters = {
            "train_batch_size": self.__config.getint(
                "train.parameters", "train_batch_size"
            ),
            "valid_batch_size": self.__config.getint(
                "train.parameters", "valid_batch_size"
            ),
            "epochs": self.__config.getint("train.parameters", "epochs"),
            "learning_rate": self.__config.getfloat(
                "train.parameters", "learning_rate"
            ),
            "shuffle": self.__config.getboolean("train.parameters", "shuffle"),
            "num_workers": self.__config.getint("train.parameters", "num_workers"),
            "max_length": self.__config.getint("MODEL", "max_length"),
        }

        self.__model = MODEL_MAP[self.__config["MODEL"]["name"]](
            load,
            save_directory,
            dataset,
            tags,
            parameters=train_parameters,
            tokenizer=self.tokenizer,
            project_name=self.__config["GENERAL"]["name"],
            pretrain=self.__config["pretrain"]["name"],
        )

    def get_tokenizer(self):
        return self.__model.tokenizer

    def get_max_length(self):
        return self.__config.getint("MODEL", "max_length")

    def run(self, data):
        output = self.__model.predict([val.ids for val in data])
        print(self.id2label)
        predictions = [[self.id2label[int(j.cpu().numpy())] for j in i] for i in output]
        if self.__config.getboolean("MODEL", "lexicon"):
            lexi_predictions = Lexicon().predict(data, self.tokenizer)
            output = Lexicon().merge(lexi_predictions, predictions)
            return output
        return predictions


if __name__ == "__main__":
    folder_path = "./data/annotated/"
    files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    manager = DatasetManager(files)

    reg = NERecognition("./src/textmining/ner/config.ini", manager)
    preprocess = Preprocess(reg.get_tokenizer(), reg.get_max_length())

    text = "Pasienten har også opplevd økt tungpust de siste månedene, noe som har begrenset aktivitetsnivået hans."

    print(reg.run(preprocess.run(text)))
