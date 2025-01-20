from transformers import BertModel
from model.nn.bilstmcrf import Model as BaseModel
from model.base.nn import NN
from structure.enum import Task


class Model(BaseModel):
    def __init__(self):
        # TODO: add config
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.word_embeds = bert_model


class BERTBiLSTMCRF:
    def __init__(
        self,
        load: bool,
        save: str,
        dataset: list = [],
        tags_name: list = [],
        parameters: dict = {},
        tokenizer=None,
        project_name: str | None = None,
    ):
        self.__model = NN(
            Model,  # type: ignore
            Task.TOKEN,
            load,
            save,
            dataset,
            tags_name,
            parameters,
            tokenizer,
            project_name,
        )
        self.tokenizer = self.__model.tokenizer

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)
