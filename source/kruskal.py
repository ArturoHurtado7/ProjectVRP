from typing import List, Tuple

class Kruskal:

    def __init__(self, edges: List[list]) -> None:
        """
        Initialize Kruskal
        """
        self.edges, self.nodes = self.code_edges(edges)
        self.partition, self.ranking, self.mst = [], [], []
        for i in range(len(self.nodes)):
            self.partition.append(i)
            self.ranking.append(1)


    def code_edges(self, edges: List[list]) -> Tuple[list, list]:
        """
        Code the edges, from discontinuous list to continuous list starting at zero
        """
        old_nodes = set()
        new_edges = []
        # get all nodes from edges and sort them
        for u,v,_ in edges:
            old_nodes.add(u)
            old_nodes.add(v)
        old_nodes = list(old_nodes)
        # coded nodes by index position in the list
        for u,v,c in edges:
            new_u = old_nodes.index(u)
            new_v = old_nodes.index(v)
            new_edges.append([new_u, new_v, c])
        # return the coded edges and original nodes
        return new_edges, old_nodes


    def decode_edges(self, edges: List[list]) -> List[list]:
        """
        decode the edges
        """
        old_edges = []
        for u,v,c in edges:
            old_edges.append([self.nodes[u], self.nodes[v], c])
        return old_edges


    def find(self, node: int) -> int:
        """
        find the root of the node
        """
        if self.partition[node] == node:
            return node
        root_node = self.find(self.partition[node])
        self.partition[node] = root_node
        return root_node


    def union(self, node_a: int, node_b: int) -> None:
        """
        union two nodes
        """
        root_a = self.find(node_a)
        root_b = self.find(node_b)
        if self.ranking[root_a] > self.ranking[root_b]:
            self.partition[root_b] = root_a
        else:
            self.partition[root_a] = root_b
            if self.ranking[root_a] == self.ranking[root_b]:
                self.ranking[root_b] += 1


    def minimum_spanning_tree(self) -> List[list]:
        """
        returns the minimum spanning tree
        """
        # sort the edges by cost
        edges = sorted(self.edges, key=lambda x: x[2])
        for u,v,c in edges:
            if self.find(u) != self.find(v):
                self.mst.append([u,v,c])
                self.union(u,v)
        return self.decode_edges(self.mst)


    def preorder(self, mst: List[list], root: int) -> List[int]:
        """
        get tree preorder path
        """
        path = [root]
        remainder = mst.copy()
        # iterate until all edges in mst are used
        while len(remainder) > 0:
            # search back to front in the preorder path
            i, remainder_nodes = -1, []
            [remainder_nodes.extend([u, v]) for v, u, _ in remainder]
            remainder_nodes = set(remainder_nodes)
            while True:
                searched_node = path[i]
                if searched_node in remainder_nodes:
                    for node_a, node_b, c in remainder:
                        if node_a == searched_node:
                            path.append(node_b)
                            remainder.remove([node_a,node_b,c])
                            break
                        if node_b == searched_node:
                            path.append(node_a)
                            remainder.remove([node_a,node_b,c])
                            break
                    break
                i -= 1
        return path

