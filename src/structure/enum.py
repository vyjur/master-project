from enum import Enum

class Dataset(Enum):
    MER = 1
    TFE = 2
    TRE = 3
    ERE = 4
    SENTENCES = 5

class Task(Enum):
    TOKEN = 1
    SEQUENCE = 2
    
class TR(Enum):
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