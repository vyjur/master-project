from typing import List
from structure.enum import TR, ME, ER

class Node:
    
    def __init__(self, value:str, type:str, context:str, date, relations: list=[], id=None):
        self.id = id
        self.value = value
        
        match type:
            case 'CONDITION':
                self.type = ME.CONDITION
            case 'SYMPTOM':
                self.type = ME.SYMPTOM
            case 'EVENT':
                self.type = ME.EVENT
            case _:
                self.type = None
        
        self.date = date
        self.context = context
        self.relations = relations
        
    def __str__(self):
        
        text = f"Node: {self.value} - {self.type} ({self.date})"
        
        for rel in self.relations:
            text += f"\n {rel}"

        return text