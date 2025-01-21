from transformers import AutoModel
from model.base.bilstm import Model as BaseModel
from model.base.nn import NN
from structure.enum import Task


class BERTBiLSTM:
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
        class Model(BaseModel):
            def __init__(self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
                BaseModel.__init__(
                    self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim
                )
                bert_model = AutoModel.from_pretrained(pretrain, trust_remote_code=True)
                self.word_embeds = bert_model

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
