# Note: this file is not used anymore
import nltk
from nltk.stem import SnowballStemmer

nltk.download("punkt")
import pandas as pd
from preprocess.setup import Preprocess


class Lexicon:
    def __init__(self):
        df = pd.read_csv("./data/NorMedTerm.csv", delimiter="\t")
        # self.lexicon = df[(df['ABBREV'] == 'CONDITION') | (df['ABBREV'] == 'PROCEDURE')]
        self.lexicon = df[(df["ABBREV"] == "CONDITION")]
        self.lexicon["ABBREV"].replace("PROCEDURE", "EVENT", inplace=True)
        self.lexicon = self.lexicon[["a.", "ABBREV"]]
        self.stemmer = SnowballStemmer("norwegian")

    def run(self, data, exact=True):
        all_labels = []
        for term in data:
            if exact:
                result = self.lexicon[
                    self.lexicon["a."].str.lower() == str(term).lower()
                ]
            else:
                result = self.lexicon[
                    self.lexicon["a."].str.contains(
                        str(term), case=False, na=False, regex=False
                    )
                ]
            if len(result) == 0:
                terms = term.split()
                if len(terms) > 0:
                    major_value = None
                    for t in terms:
                        if exact:
                            result = self.lexicon[
                                self.lexicon["a."].str.lower() == t.lower()
                            ]
                        else:
                            result = self.lexicon[
                                self.lexicon["a."].str.contains(
                                    t, case=False, na=False, regex=False
                                )
                            ]
                        if len(result) == 0:
                            continue
                        else:
                            major_value = result["ABBREV"].value_counts().idxmax()
                            all_labels.append(major_value)
                            break
                    if not major_value:
                        all_labels.append("O")
                else:
                    all_labels.append("O")
            else:
                major_value = result["ABBREV"].value_counts().idxmax()
                all_labels.append(major_value)
        return all_labels

    def predict(self, dataset, tokenizer):
        if len(dataset) == 1:
            dataset = [dataset]
        lexi_predictions = []
        for tokenized in dataset:
            words = tokenizer.decode(tokenized["input_ids"]).split()
            annot = self.run(words)
            tokens_annot = Preprocess(tokenizer).tokens_mapping(tokenized, annot)
            lexi_predictions.extend(tokens_annot)
        return lexi_predictions

    def merge(self, lexi_pred, pred):
        new_pred = list()
        for i, val in enumerate(pred):
            if val == "O" and lexi_pred[i] != "O":
                new_pred.append(lexi_pred[i])
            else:
                new_pred.append(val)
        return new_pred


if __name__ == "__main__":
    print(Lexicon().run(["hei", "på"]))

