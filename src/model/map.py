from model.nn.bert_bilstmcrf import BERTBiLSTMCRF
from model.nn.bert_bilstm import BERTBiLSTM
from model.nn.bilstmcrf import BiLSTMCRF
from model.nn.bilstm import BiLSTM
from model.base.bert import BERT
from model.bert.token_bert import TokenBERT
from model.bert.sequence_bert import SequenceBERT
from model.bert.llm import LLM

MODEL_MAP = {
    "BERT": BERT,
    "TokenBERT": TokenBERT,
    "SequenceBERT": SequenceBERT,
    "BERTBiLSTM": BERTBiLSTM,
    "BERTBiLSTMCRF": BERTBiLSTMCRF,
    "BiLSTMCRF": BiLSTMCRF,
    "BiLSTM": BiLSTM,
    "LLM": LLM,
}
