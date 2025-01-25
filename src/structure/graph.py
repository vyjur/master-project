from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        
    def dfs_enumerate_levels(self, start):
        levels = {} 
        visited = set()

        def dfs(node, current_level):
            visited.add(node)
            levels[node] = current_level 

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, current_level + 1)

        dfs(start, 0)
        return levels

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
