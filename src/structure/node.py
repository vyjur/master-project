import itertools
from datetime import datetime
from structure.enum import ME, TIMEX


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

        self.dct = dct

        self.date = date
        self.prob = 0

        self.context = context

    def __str__(self):
        text = f"Node {self.id}: {self.value} - {self.type} - ({self.date} - prob: {self.prob})"
        return text
