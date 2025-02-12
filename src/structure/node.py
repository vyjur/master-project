import itertools
from datetime import datetime
from structure.enum import ME, TR_DCT, TIMEX


class Node:
    id_iter = itertools.count()

    def __init__(
        self, value: str, type: str, dct: str | None, context: str, date: datetime
    ):
        self.id = next(self.id_iter)
        self.value = value

        self.type = None

        for me in ME:
            if type == me.name:
                self.type = me
                break
        
        for te in TIMEX:
            if type == te.name:
                self.type = te
                break

        self.dct = None
        self.set_dct(dct)

        self.date = date
        self.prob = 0
        
        self.context = context

    def __str__(self):
        text = f"Node: {self.value} - {self.type} ({self.date})"
        return text
    
    def set_dct(self, cat):
        for tr in TR_DCT:
            if cat == tr.name:
                self.dct = tr
