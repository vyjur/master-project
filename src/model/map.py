from model.bilstmcrf import BiLSTMCRF
from model.bilstm import BiLSTM
from model.bert import BERT
from model.token_bert import TokenBERT
from model.sequence_bert import SequenceBERT
from model.llm import LLM

MODEL_MAP = {
    'BERT': BERT,
    'TokenBERT': TokenBERT,
    'SequenceBERT': SequenceBERT,
    'BiLSTMCRF': BiLSTMCRF,
    'BiLSTM': BiLSTM,
    'LLM': LLM
}