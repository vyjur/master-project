from enum import Enum


class Dataset(Enum):
    NER = 1
    TRE_DCT = 2
    TRE_TLINK = 3
    SENTENCES = 4
    TEE = 5

class Task(Enum):
    TOKEN = 1
    SEQUENCE = 2

class TR_DCT(Enum):
    AFTER = 1
    BEFORE = 2
    OVERLAP = 3 
    BEFOREOVERLAP = 4

class TR_TLINK(Enum):
    BEFORE = 1
    OVERLAP = 2  
    O = 3

class TIMEX(Enum):
    DATE = 1
    DURATION = 2
    FREQUENCY = 3
    TIME = 4
    O = 5

class ME(Enum):
    CONDITION = 1
    TREATMENT = 2
    O = 3
    
class DCT(Enum):
    DATE = 1
    DCT = 2

