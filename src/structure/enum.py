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
    XAFTERY = 1
    XBEFOREY = 2
    XDURINGY = 3
    
class TR_TLINK(Enum):
    XAFTERY = 1
    XBEFOREY = 2
    XDURINGY = 3
class ME(Enum):
    CONDITION = 1
    SYMPTOM = 2
    EVENT = 3
    
class ER(Enum):
    DISEASETODISEASE = 1
    DISEASETOEVENT = 2
    DISEASETOSYMPTOM = 3
    EVENTTODISEASE = 4
    EVENTTOEVENT = 5
    EVENTTOSYMPTOM = 6
    SYMTPOMTODISEASE = 7
    SYMPTOMTOEVENT = 8
    SYMTPOMTOSYMPTOM = 9
    EQUAL = 10