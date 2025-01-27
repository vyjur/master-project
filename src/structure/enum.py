from enum import Enum


class Dataset(Enum):
    NER = 1
    TRE_DCT = 2
    TRE_TLINK = 3
    SENTENCES = 4


class Task(Enum):
    TOKEN = 1
    SEQUENCE = 2


class TR_DCT(Enum):
    AFTER = 1
    BEFORE = 2
    DURING = 3  # TODO: Change this out later
    BEFOREOVERLAP = 4


class TR_TLINK(Enum):
    AFTER = 1
    BEFORE = 2
    DURING = 3  # TODO: change this out later


class ME(Enum):
    DISEASE = 1
    SYMPTOM = 2
    TREATMENT = 3

