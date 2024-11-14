from structure.node import Node
from structure.enum import TR, ER

class Relation:
    
    def __init__(self, x: Node, y: Node, tr: str, er:str):
        self.x = x
        self.y = y
        
        match tr:
            case 'XAFTERY':
                self.tr = TR.XAFTERY
            case 'XBEFOREY':
                self.tr = TR.XBEFOREY
            case 'XDURINGY':
                self.tr = TR.XDURINGY
            case _:
                self.tr = None
                
        match er:
            case 'DISEASETODISEASE':
                self.er = ER.DISEASETODISEASE
            case 'DISEASETOEVENT':
                self.er = ER.DISEASETOEVENT
            case 'DISEASETOSYMPTOM':
                self.er = ER.DISEASETOSYMPTOM
            case 'EVENTTODISEASE':
                self.er = ER.EVENTTODISEASE
            case 'EVENTTOEVENT':
                self.er = ER.EVENTTOEVENT
            case 'EVENTTOSYMPTOM':
                self.er = ER.EVENTTOSYMPTOM
            case 'SYMPTOMTODISEASE':
                self.er = ER.SYMTPOMTODISEASE
            case 'SYMPTOMTOEVENT':
                self.er = ER.SYMPTOMTOEVENT
            case 'SYMPTOMTOSYMPTOM':
                self.er = ER.SYMTPOMTOSYMPTOM
            case 'EQUAL':
                self.er = ER.EQUAL
            case _:
                self.er = None
    
    def __str__(self):
        return f"""
            with node Y: {self.y.value} - {self.y.type}
            TR: {self.tr}
            ER: {self.er}
        """