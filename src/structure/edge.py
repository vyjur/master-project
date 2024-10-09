from structure.node import Node

class Edge:
    def __init__(self, parent: Node, child: Node):
        self.parent = parent
        self.child = child