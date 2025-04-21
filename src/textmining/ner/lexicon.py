# INFO: Baseline model. Currently fixing!
import re
import nltk
import rapidfuzz
from rapidfuzz import process

from nltk.stem import SnowballStemmer
import pandas as pd
from preprocess.setup import Preprocess
from collections import Counter

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
        
        self.__lexicon = self.lexicon.copy()


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
        predictions = []
        terms  = self.__lexicon['a.'].tolist()
        for row in data:
            
            if len(str(row)) < 3:
                predictions.append("O")
                continue
            
            threshold = 90 # quick-copy rapidfuzz.fuzz.token_sort_ratio
            #threshold = 85 quick copy 2 rapidfuzz.fuzz.token_sort_ratio

            matches = process.extract(
                row,
                terms,
                limit=5,
                scorer=rapidfuzz.fuzz.token_sort_ratio
            )

            # Filter matches by threshold and reasonable term length
            matched_values = [
                (match[0], terms.index(match[0]))
                for match in matches
                if match[1] >= threshold and len(match[0]) > 2
            ]

            # Get the ABBREVs for the matched terms
            if matched_values:
                indices = [i[1] for i in matched_values]
                instances = self.__lexicon.iloc[indices]['ABBREV'].tolist()
            else:
                instances = []
            counter = Counter(instances)
            # Get the most common element
            if len(instances):
                pred, count = counter.most_common(1)[0]
            else:
                pred = "O"
            predictions.append(pred)
        return predictions
    
    def run2(self, data):
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

    lex = Lexicon()
    text = ["hjerteinfarkt", "diabetes type 2"]

    print(lex.run(text))
