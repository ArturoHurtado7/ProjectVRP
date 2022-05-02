import math
from pyclbr import Function
from random import choice, randint, sample
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
        self.distance = self.distance_matrix(len(self.nodes))


    def distance_matrix(self, len_nodes: int) -> List[list]:
        """
        returns the distance matrix of the nodes
        """
        distance_matrix = []
        for i in range(len_nodes):
            distance_matrix.append([])
            for j in range(len_nodes):
                distance_matrix[i].append(self.euclidean_distance(i, j))
        return distance_matrix


    def nodes_edges(self, nodes: dict) -> List[list]:
        """
        Get edges of the nodes
        """
        edges = []
        keys = list(nodes.keys())
        for i, x in enumerate(keys):
            for j in range(i+1, len(keys)):
                y = keys[j]
                distance = self.distance[x][y]
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
            distance += self.distance[route[i]][route[i+1]]
        return distance


    def complete_route_distance(self, complete_route: List[List]) -> float:
        """
        returns the distance of a complete route
        """
        route = complete_route[0] + complete_route[1]
        distance = self.route_distance(route)
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
                distance_rows.append(self.distance[i][j])
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


    #--------------------------------------------------------------------------
    # Principal function
    #--------------------------------------------------------------------------
    def mt_vrpb(self) -> List[list]:
        """
        The Multiple Trip Vehicle Routing Problem with Backhauls
        """
        complete_routes = self.initial_solution()
        old_routes = [linehauls + backhauls for linehauls, backhauls in complete_routes]

        new_complete_routes = complete_routes.copy()
        last_change, i = [], 0
        while i < 10000:
            new_complete_routes, change = self.variable_neighborhood_search(new_complete_routes, self.swap_intra_route)
            if change: last_change.append([i, 'swap_intra_route'])
            new_complete_routes, change = self.variable_neighborhood_search(new_complete_routes, self.insertion_intra_route)
            if change: last_change.append([i, 'insertion_intra_route'])
            new_complete_routes, change = self.variable_neighborhood_search(new_complete_routes, self.reverse_intra_route)
            if change: last_change.append([i, 'reverse_intra_route'])
            new_complete_routes, change = self.multiple_neighborhood_search(new_complete_routes, self.insertion_inter_route)
            if change: last_change.append([i, 'insertion_inter_route'])
            new_complete_routes, change = self.multiple_neighborhood_search(new_complete_routes, self.swap_one_pair)
            if change: last_change.append([i, 'swap_one_pair'])
            new_complete_routes, change = self.multiple_neighborhood_search(new_complete_routes, self.swap_two_pairs)
            if change: last_change.append([i, 'swap_two_pairs'])
            i += 1
        print(' ********* last_change:', last_change)
        routes = [linehauls + backhauls for linehauls, backhauls in new_complete_routes]


        #print('capacity:', self.capacity)
        #print('distances: ', [self.route_distance(route) for route in routes])
        #print('demands', [self.route_demand(route) for route in routes])
        print('complete_routes: ', complete_routes)
        print('old_routes: ', old_routes)
        print('total distance: ', sum([self.route_distance(route) for route in old_routes]))
        self.plot_graph(old_routes)
        print('new_complete_routes: ', new_complete_routes)
        print('complete_routes: ', complete_routes)
        print('new_routes: ', routes)
        print('total distance: ', sum([self.route_distance(route) for route in routes]))
        self.plot_graph(routes)
        return routes


    def variable_neighborhood_search(self, routes: List[list], function: Function) -> list:
        """
        Variable neighborhood search algorithm call
        """
        change = False
        i = randint(0, len(routes) - 1) # pick a random route
        distance_before = self.complete_route_distance(routes[i])
        new_route = function(routes[i])
        distance_after = self.complete_route_distance(new_route)
        if distance_after < distance_before:
            routes[i] = new_route
            change = True
        return routes, change


    def multiple_neighborhood_search(self, routes: List[list], function: Function) -> list:
        """
        Multiple route neighborhood search algorithm call
        """
        change = False
        i, j = sample(range(len(routes)), 2)
        distance_before = self.complete_route_distance(routes[i]) + self.complete_route_distance(routes[j])
        route_i, route_j = function(routes[i], routes[j])
        distance_after = self.complete_route_distance(route_i) + self.complete_route_distance(route_j)
        if distance_after < distance_before:
            routes[i], routes[j] = route_i, route_j
            change = True
        return routes, change


    def route_ranges(self, route: list, types: int) -> list:
        """
        get the start and end index of the route without the depot
        """
        return [1, len(route)] if types == 0 else [0, len(route) - 1]


    #----------------------------------------------------------------------------------------------
    # Heuristics
    #----------------------------------------------------------------------------------------------
    def swap_intra_route(self, route: List[list]) -> list:
        """
        One swap intra route
        relocates the position of a customer at a non-adjacent arc within the same route
        taking account of the delivery customers must be served before any pickups
        """
        new_route = [x[:] for x in route]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges = self.route_ranges(new_route[types], types)
            choices = new_route[types][ranges[0]:ranges[1]]
            if len(choices) > 1:
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


    def insertion_intra_route(self, route: List[list]) -> list:
        """
        Insertion heuristic
        """
        new_route = [x[:] for x in route]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges = self.route_ranges(new_route[types], types)
            choices = new_route[types][ranges[0]:ranges[1]]
            if len(choices) > 1:
                node_a = choice(choices)
                index_a = new_route[types].index(node_a)
                choices.remove(node_a)
                node_b = choice(choices)
                index_b = new_route[types].index(node_b)
                new_route[types].pop(index_b)
                new_route[types].insert(index_a, node_b)
                break
            types = abs(types - 1)
            i += 1
        return new_route


    def reverse_intra_route(self, route: List[list]) -> list:
        """
        Reverse heuristic
        """
        new_route = [x[:] for x in route]
        types = randint(0, 1)
        ranges = self.route_ranges(new_route[types], types)
        route_part = new_route[types][ranges[0]:ranges[1]]
        route_part.reverse()
        if types == 0:
            new_route[types] = [new_route[types][0]] + route_part
        else:
            new_route[types] = route_part + [new_route[types][-1]]
        return new_route


    def insertion_inter_route(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        insert a customer from route_b into route_a
        """
        new_route_a, new_route_b = [x[:] for x in route_a], [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a = self.route_ranges(new_route_a[types], types)
            ranges_b = self.route_ranges(new_route_b[types], types)
            choices = new_route_a[types][ranges_a[0]:ranges_a[1]]
            if len(choices) > 1:
                node_a = choice(choices)
                index_b = randint(ranges_b[0], ranges_b[1])
                route_demand = self.route_demand(new_route_b[types]) + self.nodes[node_a].get('demand') 
                if route_demand < self.capacity:
                    new_route_b[types].insert(index_b, node_a)
                    new_route_a[types].remove(node_a)
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def swap_one_pair(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        swap a customer from route_a to route_b and vice versa
        """
        new_route_a, new_route_b = [x[:] for x in route_a], [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a, ranges_b = self.route_ranges(new_route_a[types], types), self.route_ranges(new_route_b[types], types)
            choices_a, choices_b = new_route_a[types][ranges_a[0]:ranges_a[1]], new_route_b[types][ranges_b[0]:ranges_b[1]]
            if len(choices_a) > 1 and len(choices_b) > 1:
                node_a, node_b = choice(choices_a), choice(choices_b)
                index_a, index_b = new_route_a[types].index(node_a), new_route_b[types].index(node_b)
                route_demand_a = self.route_demand(new_route_a[types]) - self.nodes[node_a].get('demand') + self.nodes[node_b].get('demand') 
                route_demand_b = self.route_demand(new_route_b[types]) - self.nodes[node_b].get('demand') + self.nodes[node_a].get('demand') 
                if route_demand_a < self.capacity and route_demand_b < self.capacity:
                    new_route_a[types][index_a], new_route_b[types][index_b] = node_b, node_a
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def swap_two_pairs(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        swap two customers from route_a to route_b and vice versa
        """
        new_route_a, new_route_b = [x[:] for x in route_a], [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a = self.route_ranges(new_route_a[types], types)
            ranges_b = self.route_ranges(new_route_b[types], types)
            choices_a, choices_b = new_route_a[types][ranges_a[0]:ranges_a[1]], new_route_b[types][ranges_b[0]:ranges_b[1]]
            if len(choices_a) > 2 and len(choices_b) > 2:
                node_a1, node_a2 = sample(choices_a, k=2)
                node_b1, node_b2 = sample(choices_b, k=2)
                index_a1, index_a2 = new_route_a[types].index(node_a1), new_route_a[types].index(node_a2)
                index_b1, index_b2 = new_route_b[types].index(node_b1), new_route_b[types].index(node_b2)
                nodes_demand_a = self.nodes[node_a1].get('demand') + self.nodes[node_a2].get('demand')
                nodes_demand_b = self.nodes[node_b1].get('demand') + self.nodes[node_b2].get('demand')
                route_demand_a = self.route_demand(new_route_a[types]) - nodes_demand_a + nodes_demand_b
                route_demand_b = self.route_demand(new_route_b[types]) - nodes_demand_b + nodes_demand_a
                if route_demand_a < self.capacity and route_demand_b < self.capacity:
                    new_route_a[types][index_a1], new_route_a[types][index_a2] = node_b1, node_b2
                    new_route_b[types][index_b1], new_route_b[types][index_b2] = node_a1, node_a2
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def shift_pair(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        shift a customer from route to the next route
        """
        return route_a, route_b


    def swap_single_pair(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        swap a customer to a couple of customes from route to another route
        """
        return route_a, route_b

