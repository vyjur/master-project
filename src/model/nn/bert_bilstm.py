from transformers import AutoModel
from model.nn.bilstm import Model as BaseModel
from model.base.nn import NN
from model.util import Util
from structure.enum import Task
import torch.nn as nn


class BERTBiLSTM(nn.Module):
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
        super(BERTBiLSTM, self).__init__()

        class Model(BaseModel):
            def __init__(self, batch, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
                bert_model = AutoModel.from_pretrained(pretrain, trust_remote_code=True)
                BaseModel.__init__(
                    self,
                    batch,
                    vocab_size,
                    tag_to_ix,
                    embedding_dim,
                    hidden_dim,
                    bert_model,
                )

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
            testset=testset,
        )
        self.tokenizer = self.__model.tokenizer
        self.device = self.__model.device

    def predict(self, data, pipeline=False):
        return self.__model.predict(data, pipeline)

    def forward(self, x):
        return self.__model(x)

