from model.bilstmcrf import BiLSTMCRF
from model.bert import FineTunedBert

MODEL_MAP = {
    'FineTunedBert': FineTunedBert,
    'BiLSTMCRF': BiLSTMCRF
}