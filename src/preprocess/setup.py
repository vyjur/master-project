import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
class CustomDataset(Dataset):
    def __init__(self, tokenized_data, tokenizer, label2id):
        self.tokenized_data = tokenized_data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        # Return the number of samples
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        # Get a single sample of data
        tokenized = self.tokenized_data[idx]
        tokenized_sentence = tokenized.tokens()
        labels = tokenized['labels']

        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]
        label_ids = [self.label2id[label] for label in labels]
        
        return {
              'ids': torch.tensor(ids, dtype=torch.long),
              'mask': torch.tensor(attn_mask, dtype=torch.long),
              #'token_type_ids': torch.tensor(token_ids, dtype=torch.long),
              'targets': torch.tensor(label_ids, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.tokenized_data)
class Preprocess:

    def __init__(self, tokenizer, max_length:int=512, train_size:float=0.8):
        self.__tokenizer = tokenizer
        self.__max_length = max_length
        self.__train_size = train_size

    def __tokenize_and_align_labels(self, data):
        tokenized = self.__tokenizer(data["text"], padding="max_length", max_length=self.__max_length, truncation=True, return_offsets_mapping=True)
        
        tokens = tokenized.tokens()
        offsets = tokenized["offset_mapping"]

        labels = ['O']*len(tokenized.tokens())
        for start, end, _, tag in data['entities']:
            for i, (_, (token_start, token_end)) in enumerate(zip(tokens, offsets)):
                if (token_start >= start) and (token_end <= end):
                    if token_start == start:
                        labels[i] = f"B-{tag}"
                    else:
                            labels[i] = f"I-{tag}"
        tokenized['labels'] = labels
        
        return tokenized 

    def run(self, data:list=[], tags_name:list = []):
        tags = set()
        for tag in tags_name:
            tags.add(f"B-{tag}")
            tags.add(f"I-{tag}")
                
        tags = list(tags)

        label2id = {k: v+1 for v, k in enumerate(tags)}
        id2label = {v+1: k for v, k in enumerate(tags)}

        label2id['O'] = 0
        id2label[0]='O'

        tokenized_dataset = [self.__tokenize_and_align_labels(row) for row in data]
        train, test = train_test_split(tokenized_dataset, train_size=self.__train_size, random_state=42)
        train_dataset = CustomDataset(train, self.__tokenizer, label2id)
        test_dataset = CustomDataset(test, self.__tokenizer, label2id)

        return {
            'train': train_dataset,
            'test': test_dataset,
            'label2id': label2id,
            'id2label': id2label
        }

    
if __name__ == "__main__":
    text = journal = """
        Dato: 02.10.2024
        Pasient: Kari Nordmann, f. 12.06.1980
        Henvendelse: Kontrolltime, luftveisplager
        Lege: Dr. Ola Hansen

        Anamnese:
        Pasienten kommer til kontrolltime grunnet vedvarende luftveisplager over de siste to ukene. Hun opplyser om tørrhoste, tretthet og lett feber, særlig på kveldstid. Ingen dyspné, men pasienten rapporterer om piping i brystet ved anstrengelse. Ingen kjent allergi eller astma. Ingen hjertebank eller brystsmerter.

        Tidligere sykdommer inkluderer bihulebetennelse og en enkel episode med lungebetennelse for tre år siden, behandlet med antibiotika. Pasienten jobber som lærer og har vært eksponert for flere elever med lignende symptomer.

        Aktuelt:
        Pasienten virker alment upåvirket, men sliten. Temperaturen målt hjemme er rapportert til å ligge mellom 37,5 og 38°C de siste dagene.

        Klinisk undersøkelse:
        - Allmenntilstand: Klar og orientert, noe sliten fremtoning.
        - Temperatur: 37,8°C målt ved ankomst.
        - ØNH: Rødhet i svelget, ingen belegg. Ingen forstørrede mandler. Ørene upåfallende.
        - Pulm: Normale respirasjonslyder, men svak pipelyd over basale lungefelt bilateralt ved forsert ekspirasjon. Ingen krepitasjoner eller dempning.
        - Cor: Regelmessig rytme, ingen bilyder.
        - BT: 120/80 mmHg, puls 78/min.

        Vurdering:
        Pasienten har sannsynligvis en viral luftveisinfeksjon med mulig bronkitt, uten tegn til bakteriell infeksjon eller pneumoni på nåværende tidspunkt. Mild obstruktivitet i luftveiene kan forklare piping i brystet, men ingen tegn til alvorlig astmatisk komponent.

        Plan:
        - Avvente spontan bedring, da tilstanden virker selvbegrensende.
        - Råd om å bruke reseptfrie smertestillende ved behov (Paracet 500 mg opptil x 4 daglig).
        - Vurderer å skrive ut en resept på Ventoline ved forverring av hoste eller dersom pipingen i brystet vedvarer, men anbefaler foreløpig å avvente.
        - Pasienten informeres om å kontakte lege ved forverring, dyspné, eller dersom symptomene vedvarer utover én uke.

        Kontroll: Ny time om én uke ved manglende bedring.
        """
