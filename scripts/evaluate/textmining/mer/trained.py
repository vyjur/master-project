from textmining.mer.setup import MERecognition
from preprocess.setup import Preprocess
from types import SimpleNamespace
from sklearn.metrics import classification_report

ner = MERecognition(
    config_file='./scripts/evaluate/ner/config.ini',
    manager=None,
    save_directory='./models/ner/model/b-bert'
)

preprocess = Preprocess(
    ner.get_tokenizer(), ner.get_max_length(), ner.get_stride(), ner.get_util()
)

text = 'Pasienten har vondt i ryggen siden lørdag. EKG test ble gjort ved innleggelse for å sjekke for andre problemer.'

print(ner.run(preprocess.run(text)))

text = {
    "ids": [1, 120, 4923, 120, 327, 121, 3029, 1359, 28375, 120, 124, 7910, 120, 22763, 179, 18364, 1506, 49476, 399, 1733, 10453, 3957, 186, 561, 175, 2120, 4968, 128, 2546, 29159, 3957, 5066, 126, 623, 1082, 175, 460, 1108, 561, 450, 195, 395, 175, 126, 2191, 17876, 21439, 5235, 398, 759, 558, 34158, 184, 1969, 397, 552, 1359, 1480, 24614, 498, 416, 39337, 13890, 2],
    "target": ['O', 'I-CONDITION', 'I-CONDITION', 'I-CONDITION', 'B-CONDITION', 'I-CONDITION', 'I-CONDITION', 'O', 'O', 'O', 'O', 'B-CONDITION', 'I-CONDITION', 'I-CONDITION', 'I-CONDITION', 'B-CONDITION', 'I-CONDITION', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-CONDITION', 'I-CONDITION', 'I-CONDITION', 'I-CONDITION', 'B-CONDITION', 'I-CONDITION', 'I-CONDITION', 'I-CONDITION', 'I-CONDITION', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-CONDITION', 'B-CONDITION', 'B-CONDITION', 'B-CONDITION', 'I-CONDITION', 'O']
}
output = ner.run([SimpleNamespace(**text)])
print(output[0])
print(classification_report(text['target'], output[0]))

