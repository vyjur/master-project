import torch
import torch.nn as nn
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
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

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
    def __init__(
        self,
        load: bool,
        save: str,
        dataset: list = [],
        tags_name: list = [],
        parameters: dict = [],
        tokenizer=None,
    ):
        self.__model = NN(
            Model, "sequence", load, save, dataset, tags_name, parameters, tokenizer
        )
        self.tokenizer = self.__model.tokenizer

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)
