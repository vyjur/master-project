import itertools
from datetime import datetime
from structure.enum import ME, TR_DCT


class Node:
    id_iter = itertools.count()

    def __init__(self, value: str, type: str, dct: str, context: str, date:datetime):
        self.id = next(self.id_iter)
        self.value = value

        self.type = None
        

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

        self.level = None

    def __str__(self):
        text = f"Node: {self.value} - {self.type} ({self.date})"
        return text
