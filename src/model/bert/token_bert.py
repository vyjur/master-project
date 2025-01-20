from model.base.bert import BERT
from structure.enum import Task


TASK = Task.TOKEN


class TokenBERT:
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
        self.__bert = BERT(
            TASK,
            load,
            save,
            dataset,
            tags_name,
            parameters,
            tokenizer,
            project_name,
            pretrain,
        )
        self.tokenizer = self.__bert.tokenizer

    def predict(self, data, pipeline=False):
        return self.__bert.predict(data, pipeline)
