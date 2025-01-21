import torch
import torch.nn as nn
from model.base.nn import NN
from structure.enum import Task


class Model(nn.Module):
    def __init__(
        self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bert_model=None
    ):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.batch = batch

        self.bert = bert_model is not None

        if bert_model is not None:
            self.word_embeds = bert_model
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
        else:
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
        embeds = self.word_embeds(sentences)  # type: ignore

        if self.bert:
            embeds = embeds.last_hidden_state

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
        parameters: dict = {},
        tokenizer=None,
        project_name: str | None = None,
        pretrain: str | None = None,
    ):
        self.__model = NN(
            Model,  # type: ignore
            Task.SEQUENCE,
            load,
            save,
            dataset,
            tags_name,
            parameters,
            tokenizer,
            project_name,
            pretrain,
        )
        self.tokenizer = self.__model.tokenizer

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)
