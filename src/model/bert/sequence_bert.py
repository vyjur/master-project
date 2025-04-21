import torch.nn as nn
from model.base.bert import BERT
from structure.enum import Task
from model.util import Util


TASK = Task.SEQUENCE


class SequenceBERT(nn.Module):
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
        util: Util = None,
        testset: list = [],
    ):
        super(SequenceBERT, self).__init__()

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
            util,
            testset,
        )
        self.tokenizer = self.__bert.tokenizer

    def predict(self, data, pipeline=False):
        return self.__bert.predict(data, pipeline)

    def forward(self, x):
        return self.predict(x)
