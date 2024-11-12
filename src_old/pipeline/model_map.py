from ner.bilstmcrf import BiLSTMCRF
from ner.bert import FineTunedBert
from ner.llm import LLM

MODEL_MAP = {
    'FineTunedBert': FineTunedBert,
    'BiLSTMCRF': BiLSTMCRF,
    'LLM': LLM
}