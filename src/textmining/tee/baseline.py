# INFO: Baseline model
import pandas as pd
from collections import Counter

class Baseline:
    def __init__(self):
        self.__patterns = ["Status presens", "Innlagt", "Dato", "Diktert", "Innl.dato", "Journalnotat", "utskrifts", "INNKOMSTJOURNAL"]
        self.__cities = pd.read_csv('./data/no-cities.csv')
        
    def run(self, data):
        
        for pattern in self.__patterns:
            if pattern.lower() in data[1].lower():
                return "DCT"
        
        for city in self.__cities:
            if city.lower() in data[1].lower():
                return "DCT"
        
        return "DATE"

if __name__ == "__main__":

    sentence = "Apple vurderer å kjøpe britisk oppstartfirma for en milliard dollar, men så bestemte de seg for å kjøpe en annen. Han var ikke klar for det. Hun hadde kjøpt seg et hus. Hun er kul. Hun skal klatre i morgen."            
    tee = Baseline()
    print(tee.run(sentence))