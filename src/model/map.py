from model.nn.bilstmcrf import BiLSTMCRF
from model.base.bilstm import BiLSTM
from model.base.bert import BERT
from model.bert.token_bert import TokenBERT
from model.bert.sequence_bert import SequenceBERT
from model.bert.llm import LLM

MODEL_MAP = {
    'BERT': BERT,
    'TokenBERT': TokenBERT,
    'SequenceBERT': SequenceBERT,
    'BiLSTMCRF': BiLSTMCRF,
    'BiLSTM': BiLSTM,
    'LLM': LLM
}