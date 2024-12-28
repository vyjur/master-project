import torch
import torch.nn as nn
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from preprocess.setup import Preprocess
from sklearn.metrics import accuracy_score
from model.util import Util
from model.base.nn import NN

# TODO: add weights & bias to monitor training

class Model(nn.Module):

    def __init__(self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        self.batch = batch
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

    def _get_lstm_features(self, sentences):
        embeds = self.word_embeds(sentences)
        lstm_out, (hidden, _) = self.lstm(embeds)
        lstm_out = torch.mean(lstm_out, dim=1).to(self.device)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        lstm_feats = self._get_lstm_features(sentence)
        return lstm_feats 


class BiLSTM:

    def __init__(self, load:bool, save:str, dataset: list = [], tags_name: list = [], parameters: dict = [], tokenizer = None):
        self.__model = NN(Model, 'sequence', load, save, dataset, tags_name, parameters, tokenizer)
        self.tokenizer = self.__model.tokenizer
        
    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)
    
        
if __name__ == '__main__':
    import json
    
    train_parameters = {
        'train_batch_size': 2,
        'valid_batch_size': 2,
        'epochs': 1,
        'learning_rate': 1e-04,
        'shuffle': True,
        'num_workers': 0
    }

    with open('./data/Corona2.json') as f:
        d = json.load(f)

    dataset_sample = []

    for example in d['examples']:
        
        entities = [ (annot['start'], annot['end'], annot['value'], annot['tag_name']) for annot in example['annotations']]
        
        dataset_sample.append({
            'text': example['content'],
            'entities': entities
        })

    tags = set()

    for example in d['examples']:
        for annot in example['annotations']:
            tags.add(annot['tag_name'])
            
    tags = list(tags)
    print(tags)
    
    checkpoint = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model = BiLSTM(False, dataset_sample, tags, train_parameters, tokenizer)
    tokenized = Preprocess(model.tokenizer).run([dataset_sample[0], dataset_sample[1]])
    
    pred1 = model.predict([tokenized[0]['input_ids'], tokenized[1]['input_ids']])
    pred2 = model.predict([tokenized[0]['input_ids']])
    
    print(len(tokenized[0]['input_ids']), len(pred1[0]))
    print(len(tokenized[1]['input_ids']), len(pred1[1]))
    print(len(tokenized[0]['input_ids']), len(pred2[0]))
