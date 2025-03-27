import torch
import configparser
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from model.util import Util
from structure.enum import Task, SENTENCE


class CustomDataset(Dataset):
    def __init__(self, tokenized_data, tokenizer, label2id):
        self.tokenized_data = tokenized_data
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):  # Return the number of samples
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        # Get a single sample of data
        tokenized = self.tokenized_data[idx]
        if "tokens" in tokenized:
            tokenized_sentence = tokenized["tokens"]
        else:
            tokenized_sentence = tokenized.tokens()
        labels = tokenized["labels"]
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        attn_mask = [1 if tok != "[PAD]" else 0 for tok in tokenized_sentence]
        if type(labels) is not list:
            label_ids = self.label2id[labels]
        else:
            label_ids = [self.label2id[label] for label in labels]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(attn_mask, dtype=torch.long),
            "targets": torch.tensor(label_ids, dtype=torch.long),
        }


class Preprocess:
    def __init__(self, tokenizer, max_length: int = 512, stride: int = 0, util: Util = None, train_size: float = 0.8):
        self.__tokenizer = tokenizer
        self.__util = util
        if self.__util is None:
            self.__util = Util()
        self.__max_length = max_length
        self.__stride = stride
        self.__train_size = train_size

        self.__config = configparser.ConfigParser()
        self.__config.read("./src/preprocess/config.ini")

    def run(self, data: str):
        data = str(data).split()
        tokenized_dataset = self.__tokenizer(
            data,
            padding="max_length",
            stride=self.__stride,
            max_length=self.__max_length,
            truncation=True,
            is_split_into_words=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        ).encodings
        return tokenized_dataset

    def decode(self, data: list):
        return self.__tokenizer.decode(data)

    def sliding_window(self, data, window_size=128, stride=64):
        """
        Window-size: sequence length
        stride: to create overlap
        """
        merged_dict = {}

        # Loop through each dictionary in the list
        for entry in data:
            for key, value in entry.items():
                # Append the values to the corresponding key in merged_dict
                if key in merged_dict:
                    merged_dict[key].extend(value)
                else:
                    merged_dict[key] = value

            if "tokens" in merged_dict:
                merged_dict["tokens"].extend(value)  # type: ignore
            else:
                merged_dict["tokens"] = value  # type: ignore

        windowed_sequences = []

        for i in range(0, len(merged_dict["input_ids"]), stride):
            curr_dict = {}

            for key, value in merged_dict.items():
                curr_dict[key] = merged_dict[key][i : i + window_size]
            windowed_sequences.append(curr_dict)

        return windowed_sequences

    def run_train_test_split(
        self,
        task,
        data: list = [],
        tags_name: list = [],
        window_size: int = 128,
        stride: int = 16,
    ):
        label2id, id2label = self.__util.get_tags(task, tags_name)
        print(label2id)
        print(id2label)
        tokenized_dataset = []
        
        for row in data:
            if task == Task.TOKEN:
                temp_words = row['Text'].tolist()
                temp_annot = row['MedicalEntity'].tolist()
                
                words = []
                annot = []
                terms = []
                for i, word in enumerate(temp_words):
                    for token in str(word).split():
                        words.append(token)
                        annot.append(temp_annot[i])
                        terms.append(i)
                    
                split_into_words = True
            else:
                words = row["sentence"] 
                split_into_words = False
                
            tokenized = self.__tokenizer(
                words,
                padding="max_length",
                stride=self.__stride,
                max_length=self.__max_length,
                is_split_into_words=split_into_words,
                truncation=True,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
            )
            
            included_cat = False

            for j, encoding in enumerate(tokenized.encodings):
                if task == Task.TOKEN:
                    curr_annot = [
                        annot[i] if i is not None else "O"  # type: ignore
                        for i in encoding.word_ids
                    ]
                    word_length = [
                        len(words[i]) if i is not None else None for i in encoding.word_ids
                    ]
                    curr_terms = [
                        terms[i] if i is not None else None for i in encoding.word_ids
                    ]
                    tokens_annot = self.__util.tokens_mapping(encoding, curr_annot, word_length, curr_terms)
                elif task == Task.SEQUENCE and j != 0:
                    break
                encoding_dict = {
                    "ids": encoding.ids,
                    "type_ids": encoding.type_ids,
                    "tokens": encoding.tokens,
                    "offsets": encoding.offsets,
                    "attention_mask": encoding.attention_mask,
                    "special_tokens_mask": encoding.special_tokens_mask,
                    "overflowing": encoding.overflowing,
                }
                encoding_dict["words"] = words
                if task == Task.TOKEN:
                    encoding_dict["labels"] = tokens_annot  # type: ignore
                else:
                    encoding_dict["labels"] = row["relation"]
                    if "cat" in row:
                        encoding_dict["cat"] = row["cat"]
                        included_cat = True
                    
                    if included_cat and "cat" not in encoding_dict:
                        encoding_dict['cat'] = None
                        
                tokenized_dataset.append(encoding_dict)
                
        if len(tokenized_dataset) == 0:
            return {
                "label2id": label2id,
                "id2label": id2label,
            }

        train, test = train_test_split(
            tokenized_dataset, train_size=self.__train_size, random_state=42
        )
        
        train, val = train_test_split(
            train, train_size=self.__train_size, random_state=42
        )
        
        if self.__config.getboolean("main", "window"):
            train = self.sliding_window(train, window_size=window_size, stride=stride)

        train_dataset = CustomDataset(train, self.__tokenizer, label2id)
        valid_dataset = CustomDataset(val, self.__tokenizer, label2id)
        test_dataset = CustomDataset(test, self.__tokenizer, label2id)
        
        #torch.save(test_dataset, "./data/helsearkiv/test_dataset/test_dataset.pth")
        
        return_dict = {
            "train_raw": train,
            "val_raw": val,
            "test_raw": test,
            "dataset": tokenized_dataset,
            "train": train_dataset,
            "valid": valid_dataset,
            "test": test_dataset,
            "label2id": label2id,
            "id2label": id2label,
        }
        
        if "cat" in tokenized_dataset[0]:
            intra = [d for d in test_dataset if d["cat"] == SENTENCE.INTRA]
            inter = [d for d in test_dataset if d["cat"] == SENTENCE.INTER]

            intra_dataset = CustomDataset(intra, self.__tokenizer, label2id)
            inter_dataset = CustomDataset(inter, self.__tokenizer, label2id)
            
            return_dict["intra"] = intra_dataset
            return_dict["inter"] = inter_dataset

        raise ValueError
        return return_dict


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
