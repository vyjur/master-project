# INFO: Baseline model. Currently fixing!
import re
import nltk
from nltk.stem import SnowballStemmer
import pandas as pd
from preprocess.setup import Preprocess

nltk.download("punkt")


class Lexicon:
    def __init__(self):
        self.stemmer = SnowballStemmer("norwegian")

        ### Import dictionary data
        df = pd.read_csv("./data/NorMedTerm.csv", delimiter="\t")
        self.lexicon = df[["a.", "ABBREV"]]
        self.lexicon["ABBREV"] = self.lexicon["ABBREV"].apply(  # type: ignore
            lambda x: "TREATMENT"
            if x in ["PROCEDURE", "SUBSTANCE"]
            else "CONDITION"
            if x == "CONDITION"
            else "O"
        )
        self.lexicon["Length"] = self.lexicon["ABBREV"].apply(lambda x: len(x.split()))  # type:ignore
        self.lexicon = self.lexicon[self.lexicon["Length"] < 4]

        temp = self.lexicon[["a.", "ABBREV"]].values  # type: ignore

        words = {}

        self.pattern = r"[`,~,!@#$%^&*\(\)_\-\+=\{\[\}\]\|\\:;\"'<,>\.\?/]+"
        for item in temp:
            curr = item[0].split()
            for word in curr:
                word = re.sub(self.pattern, "", word)
                word = self.stemmer.stem(word)
                if word not in words:
                    words[word] = {}
                    words[word]["Count"] = 0
                    words[word]["Category"] = set()
                words[word]["Category"].add(item[1])
                words[word]["Count"] += 1

        temp_df = pd.DataFrame(columns=["Word", "Category", "Count"])  # type: ignore
        temp_df["Word"] = words.keys()
        temp_df["Category"] = [
            list(words[word]["Category"])[0]
            if len(words[word]["Category"]) == 1
            else "O"
            for word in words
        ]
        temp_df["Count"] = [words[word]["Count"] for word in words]

        # Sort by count (optional)
        temp_df = temp_df.sort_values(by="Count", ascending=False)
        temp_df["Word-Length"] = temp_df["Word"].apply(lambda x: len(x))
        temp_df["Word"] = temp_df["Word"].apply(lambda x: self.stemmer.stem(x.lower()))

        self.lexicon = temp_df[temp_df["Word-Length"] > 2]

        common = pd.read_csv("./data/common.csv", header=None, delimiter=",")
        self.common = common.values
        self.common = [self.stemmer.stem(str(word[0])) for word in self.common]

    def run(self, data):
        words = data.strip().split()

        print("RUN", len(words))
        predictions = []

        for word in words:
            word = re.sub(self.pattern, "", word)
            word = self.stemmer.stem(str(word).lower())
            if word in self.common:
                predictions.append("O")
            else:
                result = self.lexicon[self.lexicon["Word"] == word]
                if len(result) < 1:
                    predictions.append("O")
                else:
                    predictions.append(result.to_numpy()[0][1])  # type: ignore

        return predictions


if __name__ == "__main__":
    import os
    from preprocess.dataset import DatasetManager
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from structure.enum import Dataset

    folder_path = "./data/annotated/"
    files = [
        folder_path + f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    manager = DatasetManager(files)
    raw_dataset = manager.get(Dataset.NER)

    dataset = []
    tags = set()
    for doc in raw_dataset:
        curr_doc = []
        for row in doc.itertuples(index=False):
            curr_doc.append((row[2], row[3]))  # Add (row[1], row[2]) tuple to list
            tags.add(row[3])  # Add row[2] to the set

        dataset.append(curr_doc)

    train, test = train_test_split(
        dataset,
        train_size=0.8,
        random_state=42,
    )

    print(test)

    sentences = []

    for sentence in test:
        curr_sentence = []
        target = []
        for term in sentence:
            splitted_term = term[0].split()
            for _ in range(len(splitted_term)):
                target.append(term[1])
            curr_sentence.extend(splitted_term)
        sentences.append(
            {
                "sentence": " ".join(curr_sentence),
                "words": curr_sentence,
                "target": target,
            }
        )

    print("TARGET", len(sentences[0]["target"]))

    lex = Lexicon()

    result = lex.run(sentences[0]["sentence"])

    for i, cat in enumerate(result):
        print(cat, sentences[0]["words"][i], sentences[0]["target"][i])

    target = [
        targ.replace("SYMPTOM", "CONDITION").replace("EVENT", "TREATMENT")
        for targ in sentences[0]["target"]
    ]
    print(classification_report(target, result, labels=list(tags)))

    text = "Pasienten har også opplevd økt tungpust de siste månedene, noe som har begrenset aktivitetsnivået hans og hadde hjerteinfarkt ifjor."

    print(lex.run(text))
