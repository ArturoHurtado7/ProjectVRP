
class Kruskal:

    def __init__(self, edges):
        """
        Initialize Kruskal
        """
        self.edges, self.nodes = self.code_edges(edges)
        self.partition, self.ranking, self.mst = [], [], []
        for i in range(len(self.nodes)):
            self.partition.append(i)
            self.ranking.append(1)


    def code_edges(self, edges):
        """
        Code the edges
        """
        new_nodes = set()
        new_edges = []
        for u,v,_ in edges:
            new_nodes.add(u)
            new_nodes.add(v)
        new_nodes = list(new_nodes)
        for u,v,c in edges:
            new_u = new_nodes.index(u)
            new_v = new_nodes.index(v)
            new_edges.append([new_u, new_v, c])
        return new_edges, new_nodes


    def decode_edges(self, edges):
        """
        decode the edges
        """
        new_edges = []
        for u,v,c in edges:
            new_edges.append([self.nodes[u], self.nodes[v], c])
        return new_edges


    def find(self, node):
        """
        find the root of the node
        """
        if self.partition[node] == node:
            return node
        root_node = self.find(self.partition[node])
        self.partition[node] = root_node
        return root_node


    def union(self, node1, node2):
        """
        union two nodes
        """
        root_node1 = self.find(node1)
        root_node2 = self.find(node2)
        if self.ranking[root_node1] > self.ranking[root_node2]:
            self.partition[root_node2] = root_node1
        else:
            self.partition[root_node1] = root_node2
            if self.ranking[root_node1] == self.ranking[root_node2]:
                self.ranking[root_node2] += 1


    def minimum_spanning_tree(self):
        """
        returns the minimum spanning tree
        """
        edges = sorted(self.edges, key=lambda x: x[2])
        for u,v,c in edges:
            if self.find(u) != self.find(v):
                self.mst.append([u,v,c])
                self.union(u,v)
        return self.decode_edges(self.mst)


    def preorder(self, edges, root):
        """
        get tree preorder path
        """
        path = [root]
        remainder = edges.copy()
        while len(remainder) > 0:
            index, items = 0, []
            [items.extend(item) for item in remainder]
            items = set(items)
            while True:
                index -= 1
                search = path[index]
                if search in items:
                    for x, y, c in remainder:
                        if x == search:
                            path.append(y)
                            remainder.remove([x,y,c])
                            break
                        if y == search:
                            path.append(x)
                            remainder.remove([x,y,c])
                            break
                    break
        return path

