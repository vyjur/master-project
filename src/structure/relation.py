from structure.node import Node
from structure.enum import TLINK


class Relation:
    def __init__(self, x: Node, y: Node, tr: str, prob:float):
        self.x = x
        self.y = y
        self.prob = prob
        
        self.tr = None
        
        for curr_tr in TLINK:
            if tr == curr_tr.name:
                self.tr = curr_tr
                break
        

    def __str__(self):
        return f"""
            Relation:
            Node X: {self.x.value} - {self.x.type}
            Node Y: {self.y.value} - {self.y.type}
            TLINK: {self.tr}, Probability: {self.prob}
        """

