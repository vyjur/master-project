from enum import Enum


class Dataset(Enum):
    NER = 1
    DTR = 2
    TLINK = 3
    SENTENCES = 4
    TEE = 5

class Task(Enum):
    TOKEN = 1
    SEQUENCE = 2

class DocTimeRel(Enum):
    AFTER = 1
    BEFORE = 2
    OVERLAP = 3 
    BEFOREOVERLAP = 4

class TLINK(Enum):
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

class NER_SCHEMA(Enum):
    BIO = 1
    IO = 2
    IOE = 3

class SENTENCE(Enum):
    INTER = 1
    INTRA = 2
    
class TAGS:
    XML = 1
    XML_TYPE = 2
    SOURCE = 3
    CUSTOM = 4
    
class TLINK_INPUT:
    SEP = 1
    DIST = 2