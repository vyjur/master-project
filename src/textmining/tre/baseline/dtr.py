# INFO: Baseline model
import spacy
from collections import Counter


class Baseline:
    def __init__(self):
        self.__nlp = spacy.load("nb_core_news_sm")

    def run(self, data):
        doc = self.__nlp(data)

        counts = []
        for token in doc:
            if token.pos_ in ["VERB", "AUX"]:
                if token.morph.get("Tense"):
                    counts.append(token.morph.get("Tense")[0])

                elif token.morph.get("VerbForm"):
                    if token.morph.get("VerbForm")[0] == "Part":
                        counts.append("Past")

        if len(counts) > 1:
            most_common = Counter(counts).most_common(1)[0][0]

            match most_common:
                case "Pres":
                    return "OVERLAP"
                case "Past":
                    return "BEFORE"
                case "Fut":
                    return "AFTER"
                case _:
                    return "OVERLAP"

        else:
            return "OVERLAP"


if __name__ == "__main__":
    nlp = spacy.load("nb_core_news_sm")
    sentence = "Hun skal komme tilbake om 3 uker"
    tee = Baseline()
    print(tee.run(sentence))
