from model.bilstmcrf import BiLSTMCRF
from model.bert import BERT
from model.llm import LLM

MODEL_MAP = {
    'BERT': BERT,
    'BiLSTMCRF': BiLSTMCRF,
    'LLM': LLM
}