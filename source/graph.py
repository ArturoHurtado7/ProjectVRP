import math
from matplotlib import pyplot as plt
from kruskal import Kruskal

class Graph():

    def __init__(self, nodes, capacity) -> None:
        """ 
        initializes a graph object
        """
        self.capacity = capacity
        self.nodes = dict(nodes)


    def nodes_edges(self, nodes):
        """
        Get edges of the nodes
        """
        edges = []
        keys = list(nodes.keys())
        for i, x in enumerate(keys):
            for j in range(i+1, len(keys)):
                y = keys[j]
                distance = self.euclidean_distance(x, y)
                edges.append([x, y, distance])
        return edges


    def type_nodes(self, type_node: str) -> dict:
        """
        Get nodes of a specific type
        """
        nodes = {}
        for key, value in dict(self.nodes).items():
            if value['type_node'] == type_node:
                nodes[key] = value
        return nodes


    def nodes_route(self, route: list) -> list:
        """
        Return nodes from route
        """
        nodes = {}
        for key in route:
            nodes[key] = self.nodes[key]
        return nodes


    def linehaul_nodes(self):
        """
        Get linehaul nodes
        """
        return self.type_nodes('L')
    

    def backhaul_nodes(self):
        """
        Get backhaul nodes
        """
        return self.type_nodes('B')


    def euclidean_distance(self, x: int, y: int) -> float:
        """
        returns the euclidean distance between two nodes
        """
        # get coordinates
        x1, y1 = self.nodes[x]['coordinates']
        x2, y2 = self.nodes[y]['coordinates']
        # calculate distance
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def route_distance(self, route: list) -> float:
        """
        returns the distance of a route
        """
        distance = 0
        for i in range(len(route) - 1):
            distance += self.euclidean_distance(route[i], route[i+1])
        return distance


    def route_demand(self, route) -> int:
        """
        returns the demand of a route
        """
        demand = 0
        for r in route:
            demand += self.nodes[r]['demand']
        return demand


    def polar_coordinates(self, node, initial_node = 0) -> tuple:
        """
        returns the distance and angle between depot or initial_node and selected node
        """
        # get coordinates
        x1, y1 = self.nodes[initial_node]['coordinates']
        x2, y2 = self.nodes[node]['coordinates']
        # calculate distance and angle
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = angle if angle >= 0 else 360 + angle
        return (distance, angle)


    def plot_lists(self) -> tuple:
        """
        Return the lists of coordinates, names and colors to plot
        """
        x, y, n, c = [], [], [], []
        for key, value in dict(self.nodes).items():
            x.append(int(value['coordinates'][0]))
            y.append(int(value['coordinates'][1]))
            n.append(key)
            c.append('c' if value['type_node'] == 'L' else 'r' if value['type_node'] == 'B' else 'g')
        return (x, y, n, c)


    def plot_graph(self, routes) -> None:
        """
        plots the graph with the routes
        """
        x, y, n, c = self.plot_lists()
        _, ax = plt.subplots()
        # plot nodes
        ax.scatter(x, y, marker='o', c=c, edgecolor='b')
        # plot annotations
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
        # plot routes
        if routes:
            for route in routes:
                x_route, y_route = [], []
                for item in route:
                    x_route.append(self.nodes[item]['coordinates'][0])
                    y_route.append(self.nodes[item]['coordinates'][1])
                plt.plot(x_route, y_route, 'b')
        plt.show()


    def initial_solution(self):
        """
        Initial solution with a greedy algorithm
        """
        backhaul_nodes = self.backhaul_nodes()
        linehaul_nodes = self.linehaul_nodes()
        backhauls = self.sweep_algorithm(backhaul_nodes)
        linehauls = self.sweep_algorithm(linehaul_nodes)
        while len(linehauls) < len(backhauls):
            i = -1
            while len(linehauls[i]) <= 1: i -= 1
            linehauls = [[0, linehauls[i].pop()]] + linehauls
        end_linehauls = [linehaul[-1] for linehaul in linehauls]
        end_backhauls = [backhaul[-1] for backhaul in backhauls]
        distance_matrix = self.distance_end_nodes(end_linehauls, end_backhauls)
        assignment = self.optimal_assignment(distance_matrix)
        routes = []
        for u, v in assignment:
            backhauls[v].reverse()
            routes.append(linehauls[u] + backhauls[v])

        # print results
        print('routes:', routes)
        print('capacity:', self.capacity)
        print('clusters: ', routes)
        print('distances: ', [self.route_distance(route) for route in routes])
        print('total distance: ', sum([self.route_distance(route) for route in routes]))
        print('demands', [self.route_demand(route) for route in routes])
        self.plot_graph(routes)

        return routes


    def distance_end_nodes(self, end_linehauls, end_backhauls):
        """
        Calculate the distance between the end nodes of the linehauls and backhauls
        """
        distance_matrix = []
        for i in end_linehauls:
            distance_rows = []
            for j in end_backhauls:
                distance_rows.append(self.euclidean_distance(i, j))
            distance_matrix.append(distance_rows)
        return distance_matrix
    

    def optimal_assignment(self, distance_matrix):
        recursive, _ = self.recursive_assignment(distance_matrix, 0, [])
        assignment = []
        for row in recursive: 
            assignment += row
        assignment = sorted(assignment, key=lambda x: x[0])
        optimal, i = [], 0
        while assignment[i][0] == assignment[0][0]:
            if (assignment[i][1][0] not in [o[0] for o in optimal] and
                assignment[i][1][1] not in [o[1] for o in optimal]):
                optimal.append(assignment[i][1])
            i += 1
        return optimal


    def recursive_assignment(self, distance_matrix: list, accumulated_distance: int, route: list) -> tuple:
        """
        Get the optimal assignment by recursively adding the closest node to the route
        """
        answer = []
        if len(distance_matrix) == 1:
            return accumulated_distance + distance_matrix[0][0], route
        for i, rows in enumerate(distance_matrix):
            for j, distance in enumerate(rows):
                d_matrix = [[i for x, i in enumerate(row) if x!= j] for y, row in enumerate(distance_matrix) if y!= i]
                d_distance = accumulated_distance + distance
                d_route = route if route else [i, j]
                r_matrix, r_route = self.recursive_assignment(d_matrix, d_distance, d_route)
                if r_route:
                    answer.append([r_matrix, r_route])
                else:
                    answer.append(r_matrix)
        return answer, []


    def sweep_algorithm(self, nodes):
        """
        Initial solution with a greedy algorithm
        """
        items, routes = [], []
        # Get nodes polar coordinates and demand
        for key in nodes.keys():
            _, angle, = self.polar_coordinates(key)
            items.append([key, angle])
        # Create routes
        sorted_items = sorted(items, key=lambda x: x[1])
        clusters = self.create_clusters(sorted_items)
        routes = self.create_routes(clusters)
        return routes


    def create_clusters(self, sorted_nodes) -> list:
        """
        Create clusters from sorted nodes with capacity restrictions
        """
        demand, cluster, clusters = 0, [0], []
        for node in sorted_nodes:
            node_demand = self.nodes[node[0]]['demand']
            if demand + node_demand > self.capacity:
                clusters.append(cluster)
                demand, cluster = 0, [0]
            cluster.append(node[0])
            demand += node_demand
        if demand > 0:
            clusters.append(cluster)
        return clusters


    def create_routes(self, clusters):
        """
        Create routes from clusters
        """
        routes = []
        for cluster in clusters:
            cluster_nodes = self.nodes_route(cluster)
            cluster_edges = self.nodes_edges(cluster_nodes)
            kruskal = Kruskal(cluster_edges)
            spanning_tree = kruskal.minimum_spanning_tree()
            preorder = kruskal.preorder(spanning_tree, 0)
            routes.append(preorder)
        return routes



    # Heuristis
    def insertion_intra_route(self, route) -> list:
        pass


    def insertion_inter_route(self, route1, route2) -> tuple:
        pass


    def swap_one_pair(self, route1, route2) -> tuple:
        pass
    
    
    def swap_two_pairs(self, route1, route2) -> tuple:
        pass
    
    
    def shift_none_pair(self, route):
        pass


    def single_pair_swap(self, route):
        pass

