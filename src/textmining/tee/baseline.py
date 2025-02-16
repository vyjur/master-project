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
                    return 'DCT'
                case _:
                    return 'DATE'
                
        else:
            return 'DATE'


if __name__ == "__main__":

    nlp = spacy.load("nb_core_news_sm")
    sentence = "Apple vurderer å kjøpe britisk oppstartfirma for en milliard dollar, men så bestemte de seg for å kjøpe en annen. Han var ikke klar for det. Hun hadde kjøpt seg et hus. Hun er kul. Hun skal klatre i morgen."            
    tee = Baseline()
    print(tee.run(sentence))