# INFO: Baseline model. Currently fixing!
import nltk
from collections import Counter
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
        self.lexicon["ABBREV"] = self.lexicon["ABBREV"].apply(lambda x: 'TREATMENT' if x in ['PROCEDURE', 'SUBSTANCE'] 
                                   else 'CONDITION' if x == 'CONDITION' 
                                   else 'O')
        self.lexicon["Length"] = self.lexicon["ABBREV"].apply(lambda x: len(x.split()))
        self.lexicon = self.lexicon[self.lexicon["Length"] < 4]
        
        temp = self.lexicon[['a.', "ABBREV"]].values

        words = {   
}
        for item in temp:
            curr = item[0].split()
            for word in curr:
                word = self.stemmer.stem(word)
                if word not in words:
                    words[word] = {}
                    words[word]['Count'] = 0
                    words[word]['Category'] = set()
                words[word]['Category'].add(item[1])
                words[word]['Count'] += 1

        temp_df = pd.DataFrame(columns=['Word', 'Category', 'Count'])
        temp_df['Word'] = words.keys()
        temp_df['Category'] = [ list(words[word]['Category'])[0] if len(words[word]['Category']) == 1 else 'O' for word in words]
        temp_df['Count'] = [words[word]['Count'] for word in words]

        # Sort by count (optional)
        temp_df = temp_df.sort_values(by='Count', ascending=False)
        temp_df['Word-Length']= temp_df['Word'].apply(lambda x: len(x))
        temp_df['Word'] = temp_df['Word'].apply(lambda x: self.stemmer.stem(x.lower()))

        self.lexicon = temp_df[temp_df["Word-Length"] > 2] 
        
        common = pd.read_csv("./data/common.csv", header=None, delimiter=",")
        self.common = common.values
        self.common = [self.stemmer.stem(str(word[0])) for word in self.common]
    
    def run(self, data):
        words = data.strip().split()
        
        predictions = []
        
        for word in words:
            word = self.stemmer.stem(str(word).lower())
            if word in self.common:
                predictions.append("O")
            else:
                result = self.lexicon[self.lexicon['Word'] == word]
                if len(result) < 1:
                    predictions.append("O")
                else:
                    predictions.append(result.to_numpy()[0][1])
        
        return predictions
            

if __name__ == "__main__":
    lex = Lexicon()
    text = "Pasienten har også opplevd økt tungpust de siste månedene, noe som har begrenset aktivitetsnivået hans og hadde hjerteinfarkt ifjor."

    print(lex.run(text))
