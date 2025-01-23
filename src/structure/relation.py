from structure.node import Node
from structure.enum import TR_TLINK, ER


class Relation:
    def __init__(self, x: Node, y: Node, tr: str, er: str):
        self.x = x
        self.y = y
        
        for curr_tr in TR_TLINK:
            if tr == curr_tr.name:
                self.tr = curr_tr
                break

    def __str__(self):
        return f"""
            with node Y: {self.y.value} - {self.y.type}
            TR_TLINK: {self.tr}
        """

