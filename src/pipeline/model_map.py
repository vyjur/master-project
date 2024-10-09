from ner.bilstmcrf import BiLSTMCRF
from ner.bert import FineTunedBert

MODEL_MAP = {
    'FineTunedBert': FineTunedBert,
    'BiLSTMCRF': BiLSTMCRF
}