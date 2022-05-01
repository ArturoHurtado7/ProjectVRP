import math
from pyclbr import Function
from random import choice, randint
from matplotlib import pyplot as plt
from kruskal import Kruskal
from typing import List, Tuple

class Graph():

    def __init__(self, nodes: dict, capacity: int, vehicles: int, maximun_driving: int) -> None:
        """ 
        initializes a graph object
        """
        self.capacity = capacity
        self.nodes = nodes
        self.backhauls = self.backhaul_nodes(self.nodes)
        self.linehauls = self.linehaul_nodes(self.nodes)
        self.vehicles = vehicles
        self.maximun_driving = maximun_driving


    def nodes_edges(self, nodes: dict) -> List[list]:
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


    def type_nodes(self, type_node: str, nodes: dict) -> dict:
        """
        Get nodes of a specific type
        """
        typed_nodes = {}
        for key, value in nodes.items():
            if value['node_type'] == type_node:
                typed_nodes[key] = value
        return typed_nodes


    def nodes_route(self, route: list) -> dict:
        """
        Return nodes from route
        """
        nodes = {}
        for key in route:
            nodes[key] = self.nodes[key]
        return nodes


    def linehaul_nodes(self, nodes: dict) -> dict:
        """
        Get linehaul nodes
        """
        return self.type_nodes('L', nodes)
    

    def backhaul_nodes(self, nodes: dict) -> dict:
        """
        Get backhaul nodes
        """
        return self.type_nodes('B', nodes)


    def euclidean_distance(self, node_a: int, node_b: int) -> float:
        """
        returns the euclidean distance between two nodes
        """
        # get coordinates
        x1, y1 = self.nodes[node_a]['coordinates']
        x2, y2 = self.nodes[node_b]['coordinates']
        # calculate distance
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def route_distance(self, route: List[int]) -> float:
        """
        returns the distance of a route
        """
        distance = 0
        for i in range(len(route) - 1):
            distance += self.euclidean_distance(route[i], route[i+1])
        return distance


    def route_demand(self, route: List[int]) -> int:
        """
        returns the demand of a route
        """
        demand = 0
        for r in route:
            demand += self.nodes[r]['demand']
        return demand


    def polar_coordinates(self, node: int, initial_node: int = 0) -> tuple:
        """
        returns the distance and angle between depot or initial_node and selected node
        """
        # get coordinates
        x1, y1 = self.nodes[initial_node]['coordinates']
        x2, y2 = self.nodes[node]['coordinates']
        # calculate distance and angle
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        #angle = angle if angle >= 0 else 360 + angle
        return (distance, angle)


    def plot_lists(self) -> Tuple[list, list, list, list]:
        """
        Return the lists of coordinates, names and colors to plot
        """
        x, y, n, c = [], [], [], []
        for key, value in dict(self.nodes).items():
            x.append(int(value['coordinates'][0]))
            y.append(int(value['coordinates'][1]))
            n.append(key)
            c.append('c' if value['node_type'] == 'L' else 'r' if value['node_type'] == 'B' else 'g')
        return (x, y, n, c)


    def plot_graph(self, routes: List[list]) -> None:
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
        # apply greedy algorithm for linehauls and backhauls
        backhauls = self.sweep_algorithm(self.backhauls)
        linehauls = self.sweep_algorithm(self.linehauls)
        # ensure that the number of linehauls is equal to the number of backhauls
        while len(linehauls) < len(backhauls):
            i = -1
            while len(linehauls[i]) <= 1: i -= 1
            linehauls = [[0, linehauls[i].pop()]] + linehauls
        while len(linehauls) > len(backhauls):
            backhauls.append([0])
        # get all the end nodes
        end_linehauls = [linehaul[-1] for linehaul in linehauls]
        end_backhauls = [backhaul[-1] for backhaul in backhauls]
        # get all the distances between all the end nodes
        distance_matrix = self.distance_end_nodes(end_linehauls, end_backhauls)
        # get the route with the minimum distance connecting the end nodes
        assignment = self.optimal_assignment(distance_matrix)
        # combine the routes between linehauls and backhauls
        routes = []
        for u, v in assignment:
            backhauls[v].reverse()
            routes.append([linehauls[u], backhauls[v]])
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
        # sort by distance
        assignment = sorted(assignment, key=lambda x: x[0])
        optimal, i = [], 0
        # accumulate the positions of the less distance routes
        while assignment[i][0] == assignment[0][0]:
            if (assignment[i][1][0] not in [o[0] for o in optimal] and
                assignment[i][1][1] not in [o[1] for o in optimal]):
                optimal.append(assignment[i][1])
            i += 1
        # return the optimal assignment positions
        return optimal


    def recursive_assignment(self, distance_matrix: list, accumulated_distance: int, route: list) -> Tuple[List[list], List]:
        """
        Get the optimal assignment by recursively adding the closest node to the route
        """
        answer = []
        # end condition, when the distance matrix is only one row
        if len(distance_matrix) == 1:
            return accumulated_distance + distance_matrix[0][0], route
        # iterate over the rows of the distance matrix
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


    def sweep_algorithm(self, nodes: dict) -> list:
        """
        Initial solution with a greedy algorithm
        """
        items, routes = [], []
        # Get node's polar coordinates
        for key in nodes.keys():
            _, angle, = self.polar_coordinates(key)
            items.append([key, angle])
        # Create routes of nodes by sorted angles
        sorted_items = sorted(items, key=lambda x: x[1])
        clusters = self.create_clusters(sorted_items)
        routes = self.create_routes(clusters)
        return routes


    def create_clusters(self, sorted_nodes: List[list]) -> list:
        """
        Create clusters from sorted nodes with capacity restrictions
        """
        demand, cluster, clusters = 0, [0], []
        # Create clusters from sorted nodes
        for node in sorted_nodes:
            key = node[0]
            node_demand = self.nodes[key].get('demand')
            if demand + node_demand > self.capacity:
                clusters.append(cluster)
                demand, cluster = 0, [0]
            cluster.append(key)
            demand += node_demand
        # add last cluster if it is not empty
        if demand > 0:
            clusters.append(cluster)
        return clusters


    def create_routes(self, clusters: List[list]) -> list:
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

    def mt_vrpb(self) -> List[list]:
        """
        The Multiple Trip Vehicle Routing Problem with Backhauls
        """
        complete_routes = self.initial_solution()
        print('complete_routes: ', complete_routes)
        routes = [linehauls + backhauls for linehauls, backhauls in complete_routes]
        print('routes: ', routes)
        #print('capacity:', self.capacity)
        #print('distances: ', [self.route_distance(route) for route in routes])
        print('total distance: ', sum([self.route_distance(route) for route in routes]))
        #print('demands', [self.route_demand(route) for route in routes])
        #self.plot_graph(routes)
        c_routes = complete_routes.copy()
        new_complete_routes = self.variable_neighborhood_search(c_routes, self.insertion_intra_route)
        print('new_complete_routes: ', new_complete_routes)
        print('complete_routes: ', complete_routes)
        routes = [linehauls + backhauls for linehauls, backhauls in new_complete_routes]
        print('routes: ', routes)
        print('total distance: ', sum([self.route_distance(route) for route in routes]))
        #self.plot_graph(routes)
        return routes


    def variable_neighborhood_search(self, routes: List[list], function: Function) -> list:
        """
        Variable neighborhood search algorithm call
        """
        i = randint(0, len(routes) - 1) # pick a random route
        distance_before = self.route_distance(routes[i][0] + routes[i][1])
        new_route = function(routes[i])
        distance_after = self.route_distance(new_route[0] + new_route[1])
        if distance_after < distance_before:
            routes[i] = new_route
        return routes

    #----------------------------------------------------------------------------------------------
    # Heuristics
    #----------------------------------------------------------------------------------------------
    def insertion_intra_route(self, route: List[list]) -> list:
        """
        One insertion intra route
        relocates the position of a customer at a non-adjacent arc within the same route
        taking account of the delivery customers must be served before any pickups
        """
        new_route = [x[:] for x in route]
        types = randint(0, 1) # 0: linehaul, 1: backhaul
        i = 0
        while i < 2:
            if types == 0: # linehaul -> from 1 to n
                ranges = [1, len(new_route[types])]
            else: # backhaul -> from 0 to n-1
                ranges = [0, len(new_route[types])-1]
            choices = new_route[types][ranges[0]:ranges[1]]
            if len(choices) > 2:
                node_a = choice(choices)
                index_a = new_route[types].index(node_a)
                choices.remove(node_a)
                node_b = choice(choices)
                index_b = new_route[types].index(node_b)
                new_route[types][index_a] = node_b
                new_route[types][index_b] = node_a
                break
            types = abs(types - 1)
            i += 1
        return new_route


    def insertion_inter_route(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        
        """
        pass


    def swap_one_pair(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        
        """
        pass


    def swap_two_pairs(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        
        """
        pass


    def shift_none_pair(self, route: List[int]) -> List[int]:
        """
        
        """
        pass


    def single_pair_swap(self, route: List[int]) -> List[int]:
        """
        
        """
        pass
