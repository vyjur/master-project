class Node:
    def __init__(self, label:str, prob:float, edges:list, document:str):
        self.label = label
        self.prob = prob
        self.edges = edges
        self.document = document
