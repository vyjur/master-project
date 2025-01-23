from structure.enum import ME, TR_DCT

# TODO: change name?
class Node:
    def __init__(
        self, value: str, type: str, dct: str, context: str, date, relations: list = [], id=None
    ):
        self.id = id
        self.value = value
        
        self.type = None
        
        # TODO: should this be in here or move to pipeline? and let this be dummy class?
        for me in ME:
            if type == me.name:
                self.type = me
                break
        
        self.dct = None
        for tr in TR_DCT:
            if dct == tr.name:
                self.dct = tr
            
        self.date = date
        self.context = context
        self.relations = relations

    def __str__(self):
        text = f"Node: {self.value} - {self.type} ({self.date})"

        for rel in self.relations:
            text += f"\n {rel}"

        return text

