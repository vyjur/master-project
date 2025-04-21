from collections import defaultdict


class Graph:
    def __init__(self):
        self.graph = defaultdict(list)  # Adjacency list representation
        self.reverse_graph = defaultdict(
            list
        )  # Reverse graph to track parent-child relationships

    def add_node(self, u):
        self.graph[u] = list()

    def remove_node(self, node):
        # Get all children of the node
        children = self.graph[node]

        # Get all parents of the node (nodes that have edges pointing to it)
        parents = self.reverse_graph[node]

        # Reassign children of the removed node to its parents
        for parent in parents:
            # Remove the original connection from parent to the removed node
            if node in self.graph[parent]:
                self.graph[parent].remove(node)
            # Add all the children to the parent's adjacency list
            self.graph[parent].extend(children)

        # Remove the node from the graph
        if node in self.graph:
            del self.graph[node]  # Delete the node from the graph
        if node in self.reverse_graph:
            del self.reverse_graph[node]  # Delete the node from reverse graph

        # Remove the node from other nodes' adjacency lists
        for neighbors in self.graph.values():
            if node in neighbors:
                neighbors.remove(node)

        # Also, remove the node from reverse graph's parent-child relationships
        for children in self.reverse_graph.values():
            if node in children:
                children.remove(node)

    def add_edge(self, u, v):
        self.graph[u].append(v)  # Directed edge u -> v
        self.reverse_graph[v].append(u)  # Reverse edge v -> u (parent -> child)

    def remove_edge(self, u, v):
        if v in self.graph[u]:
            self.graph[u].remove(v)
        if u in self.reverse_graph[v]:
            self.reverse_graph[v].remove(u)

    def __is_cyclic_util(self, v, visited, recursion_stack):
        visited[v] = True
        recursion_stack[v] = True

        for neighbor in self.graph[v]:
            if neighbor in visited and not visited[neighbor]:
                if self.__is_cyclic_util(neighbor, visited, recursion_stack):
                    return True
            elif neighbor in recursion_stack and recursion_stack[neighbor]:
                return True

        recursion_stack[v] = False
        return False

    def is_cyclic(self):
        visited = {node: False for node in self.graph}
        recursion_stack = {node: False for node in self.graph}

        for node in self.graph:
            if not visited[node]:
                if self.__is_cyclic_util(node, visited, recursion_stack):
                    return True

        return False

    def enumerate_levels(self):
        levels = {}  # To store levels of each node
        visited = set()

        def dfs(node, current_level):
            visited.add(node)
            levels[node] = current_level  # Assign the current level to the node

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_level + 1)

        # Handle disconnected graph
        for node in list(
            self.graph.keys()
        ):  # Use list(self.graph.keys()) to avoid RuntimeError
            if node not in visited:
                dfs(node, 0)  # Start DFS for unvisited components

        return levels


if __name__ == "__main__":
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 0)

    if g.is_cyclic():
        print("Cycle detected")
    else:
        print("No cycle added")
