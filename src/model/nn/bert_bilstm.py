from transformers import BertModel
from model.base.bilstm import Model as BaseModel
from model.base.nn import NN
from structure.enum import Task


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
        pretrain: str | None = None,
    ):
        class Model(BaseModel):
            def __init__(self):
                bert_model = BertModel.from_pretrained(pretrain)
                self.word_embeds = bert_model

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
            pretrain,
        )
        self.tokenizer = self.__model.tokenizer

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)
