import math
from random import choice, randint, sample
from matplotlib import pyplot as plt
from kruskal import Kruskal
from typing import List, Tuple, Callable

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


    def plot_lists(self, plot_nodes) -> Tuple[list, list, list, list]:
        """
        Return the lists of coordinates, names and colors to plot
        """
        x, y, n, c, d = [], [], [], [], []
        for key, value in dict(plot_nodes).items():
            x.append(int(value['coordinates'][0]))
            y.append(int(value['coordinates'][1]))
            n.append(key)
            c.append('c' if value['node_type'] == 'L' else 'r' if value['node_type'] == 'B' else 'g')
            d.append(value['demand'])
        return (x, y, n, c, d)


    def plot_graph(self, routes: List[list], title: str = '') -> None:
        """
        plots the graph with the routes
        """
        # get the unique nodes
        p_nodes = []
        [[p_nodes.append(item) for item in route] for route in routes]
        p_nodes = list(set(p_nodes))
        plot_nodes = self.nodes_route(p_nodes)
        # get lists of coordinates, names and colors
        x, y, n, c, d = self.plot_lists(plot_nodes)
        fig, ax = plt.subplots()
        fig.suptitle(title)
        # plot routes
        for route in routes:
            x_route, y_route = [], []
            for item in route:
                x_route.append(self.nodes[item]['coordinates'][0])
                y_route.append(self.nodes[item]['coordinates'][1])
            plt.plot(x_route, y_route, 'b', linewidth=1)
        # plot nodes
        ax.scatter(x, y, marker='o', c=c, edgecolor='black', linewidth=0.5)
        # plot node names
        for i, txt in enumerate(n):
            ax.annotate(str(txt), (x[i]-0.3, y[i]+1), fontsize=12)
        # plot node demands
        for i, txt in enumerate(d):
            if i != 0: ax.annotate(str(txt), (x[i]-0.3, y[i]-3), fontsize=10, color='g')
        plt.show()
    

    def plot_graph2(self, routes_a: List[list], routes_b: List[list], title: str = '') -> None:
        """
        plots the graph with the routes
        """
        # create figure
        
        _, ax = plt.subplots(1, 2)
        ax[0].axes.xaxis.set_visible(False)
        ax[0].axes.yaxis.set_visible(False)
        ax[1].axes.xaxis.set_visible(False)
        ax[1].axes.yaxis.set_visible(False)
        #fig.suptitle(title)

        # get the unique nodes
        p_nodes = []
        [[p_nodes.append(item) for item in route] for route in routes_a]
        p_nodes = list(set(p_nodes))
        plot_nodes = self.nodes_route(p_nodes)

        # get lists of coordinates, names and colors
        x, y, n, c, d = self.plot_lists(plot_nodes)
        
        # plot routes
        for route in routes_a:
            x_route, y_route = [], []
            for item in route:
                x_route.append(self.nodes[item]['coordinates'][0])
                y_route.append(self.nodes[item]['coordinates'][1])
            ax[0].plot(x_route, y_route, 'b')
        # plot nodes
        ax[0].scatter(x, y, marker='o', c=c, edgecolor='black')
        # plot node names
        for i, txt in enumerate(n):
            ax[0].annotate(str(txt), (x[i]-0.3, y[i]+1), fontsize=18)
        
        # second graph
        p_nodes = []
        [[p_nodes.append(item) for item in route] for route in routes_b]
        p_nodes = list(set(p_nodes))
        plot_nodes = self.nodes_route(p_nodes)
        
        # get lists of coordinates, names and colors
        x, y, n, c, d = self.plot_lists(plot_nodes)

        for route in routes_b:
            x_route, y_route = [], []
            for item in route:
                x_route.append(self.nodes[item]['coordinates'][0])
                y_route.append(self.nodes[item]['coordinates'][1])
            ax[1].plot(x_route, y_route, 'b')
        # plot nodes
        ax[1].scatter(x, y, marker='o', c=c, edgecolor='black')
        # plot node names
        for i, txt in enumerate(n):
            ax[1].annotate(str(txt), (x[i]-0.3, y[i]+1), fontsize=18)

        plt.show()


    #--------------------------------------------------------------------------
    # Initial Solution Methods
    #--------------------------------------------------------------------------
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
            while len(linehauls[i]) < 3:
                i -= 1
            item = linehauls[i].pop()
            linehauls = [[0, item]] + linehauls
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


    def optimal_assignment(self, distance_matrix: list) -> List[list]:
        """
        Get the optimal assignment by recursively adding the closest node to the route
        """
        selected_columns, selected_rows, path = set(), set(), []
        while len(selected_columns) < len(distance_matrix):
            least_distance, least_coord = 0, []
            l_i, l_j = 0, 0
            for i, rows in enumerate(distance_matrix):
                for j, distance in enumerate(rows):
                    if (i not in selected_columns and j not in selected_rows and 
                       (distance < least_distance or least_distance == 0)):
                        least_distance = distance
                        l_i, l_j = i, j
                        least_coord = [i, j]
            path.append(least_coord)
            selected_columns.add(l_i)
            selected_rows.add(l_j)
        return path

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

        in_limit, out_limit = 800, 400
        new_distance, best_distance = 0, 0
        new_complete_routes, best_complete_routes = [], []
        for i in range(out_limit):
            new_complete_routes = complete_routes.copy()
            for j in range(in_limit):
                new_complete_routes = self.framework_optimizer(new_complete_routes)
            routes = [linehauls + backhauls for linehauls, backhauls in new_complete_routes]
            new_distance = sum([self.route_distance(route) for route in routes])
            if new_distance < best_distance or best_distance == 0:
                print(f'New distance: {new_distance} at {i}, {j}')
                best_distance = new_distance
                best_complete_routes = new_complete_routes

        #375.3044602493867
        #375.2797871480125
        #routes = [[0, 9, 7, 5, 2, 1, 6, 0], [0, 14, 21, 19, 16, 0], [0, 10, 8, 3, 4, 11, 13, 0], [0, 12, 15, 18, 20, 17, 0]]


        #print('capacity:', self.capacity)
        #print('distances: ', [self.route_distance(route) for route in routes])
        #print('demands', [self.route_demand(route) for route in routes])
        #print('complete_routes: ', complete_routes)
        #print('old_routes: ', old_routes)
        #print('total distance: ', sum([self.route_distance(route) for route in old_routes]))
        #self.plot_graph(old_routes)
        print('best_complete_routes: ', best_complete_routes)
        routes = [linehauls + backhauls for linehauls, backhauls in best_complete_routes]
        #print('complete_routes: ', complete_routes)
        print('new_routes: ', routes)
        print('total distance: ', sum([self.route_distance(route) for route in routes]))
        print('************************************************************************************')
        self.plot_graph(routes, 'mt_vrpb')

        #complete_routes = [[[0, 11, 3, 8], [13, 19, 16, 0]], [[0, 6, 1, 2, 5, 7], [18, 0]]]
        #old_routes = [linehauls + backhauls for linehauls, backhauls in complete_routes]
        #print('total distance: ', sum([self.route_distance(route) for route in old_routes]))

        #best_complete_routes = [[[0, 8, 3, 11], [13, 19, 16, 0]], [[0, 6, 1, 2, 5, 7], [18, 0]]]
        #routes = [linehauls + backhauls for linehauls, backhauls in best_complete_routes]
        #print('total distance: ', sum([self.route_distance(route) for route in routes]))

        #self.plot_graph2(old_routes, routes, '1-insertion')

        return routes


    def framework_optimizer(self, complete_routes: List[list]) -> List[list]:
        """
        Local search optimiser framework
        """
        complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
        complete_routes, change = self.variable_neighborhood_search(complete_routes, self.swap_intra_route)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.variable_neighborhood_search(complete_routes, self.reverse_intra_route)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.insertion_inter_route)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.swap_one_pair)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.swap_two_pairs)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.shift_pair)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.swap_single_pair)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.swap_backhauls)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.variable_neighborhood_divide(complete_routes, self.divide_route)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.multiple_neighborhood_search(complete_routes, self.shift_continuos_pair)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        complete_routes, change = self.variable_neighborhood_joint(complete_routes, self.joint_linehauls)
        if change: 
            complete_routes, change = self.variable_neighborhood_search(complete_routes, self.insertion_intra_route)
            complete_routes = self.framework_optimizer(complete_routes)
        return complete_routes


    def variable_neighborhood_search(self, routes: List[list], function: Callable) -> list:
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


    def variable_neighborhood_divide(self, routes: List[list], function: Callable) -> list:
        """
        Variable neighborhood divide route algorithm call
        """
        change = False
        i = randint(0, len(routes) - 1) # pick a random route
        distance_before = self.complete_route_distance(routes[i])
        first_route, second_route = function(routes[i])
        if second_route:
            distance_after = self.complete_route_distance(first_route) + self.complete_route_distance(second_route)
            if distance_after < distance_before:
                routes[i] = first_route
                routes.append(second_route)
                change = True
        return routes, change


    def variable_neighborhood_joint(self, routes: List[list], function: Callable) -> list:
        """
        Variable neighborhood joint route algorithm call
        """
        change = False
        i, j = sample(range(len(routes)), 2) # pick two random routes
        route_demand = self.route_demand(routes[i][0]) + self.route_demand(routes[j][0])
        if (routes[i][1] == [0] or routes[j][1] == [0]) and route_demand <= self.capacity:
            distance_before = self.complete_route_distance(routes[i]) + self.complete_route_distance(routes[j])
            route = function(routes[i], routes[j])
            distance_after = self.complete_route_distance(route)
            if distance_after < distance_before:
                max_index, min_index = max(i, j), min(i, j)
                routes[min_index] = route
                routes.remove(routes[max_index])
                change = True
        return routes, change


    def multiple_neighborhood_search(self, routes: List[list], function: Callable) -> list:
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
                if route_demand <= self.capacity:
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
                if route_demand_a <= self.capacity and route_demand_b <= self.capacity:
                    new_route_a[types][index_a], new_route_b[types][index_b] = node_b, node_a
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def swap_two_pairs(self, route_a: List[int], route_b: List[int]) -> Tuple[list, list]:
        """
        swaps two pairs of consecutive customers taken from two separate routes
        """
        new_route_a = [x[:] for x in route_a]
        new_route_b = [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a = self.route_ranges(new_route_a[types], types)
            ranges_b = self.route_ranges(new_route_b[types], types)
            choices_a = new_route_a[types][ranges_a[0]:ranges_a[1]-1]
            choices_b = new_route_b[types][ranges_b[0]:ranges_b[1]-1]
            if len(choices_a) > 2 and len(choices_b) > 2:
                # select two consecutive customers from route_a
                node_a1 = choice(choices_a)
                index_a1 = new_route_a[types].index(node_a1)
                index_a2 = index_a1 + 1
                node_a2 = new_route_a[types][index_a2]
                # select two consecutive customers from route_b
                node_b1 = choice(choices_b)
                index_b1 = new_route_b[types].index(node_b1)
                index_b2 = index_b1 + 1
                node_b2 = new_route_b[types][index_b2]
                # validate swap with demand constraints
                nodes_demand_a = self.nodes[node_a1].get('demand') + self.nodes[node_a2].get('demand')
                nodes_demand_b = self.nodes[node_b1].get('demand') + self.nodes[node_b2].get('demand')
                route_demand_a = self.route_demand(new_route_a[types]) - nodes_demand_a + nodes_demand_b
                route_demand_b = self.route_demand(new_route_b[types]) - nodes_demand_b + nodes_demand_a
                # swap two consecutive customers
                if route_demand_a <= self.capacity and route_demand_b <= self.capacity:
                    new_route_a[types][index_a1], new_route_a[types][index_a2] = node_b1, node_b2
                    new_route_b[types][index_b1], new_route_b[types][index_b2] = node_a1, node_a2
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def shift_pair(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        re-locates two consecutive customers from one route to another
        """
        new_route_a = [x[:] for x in route_a]
        new_route_b = [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a = self.route_ranges(new_route_a[types], types)
            ranges_b = self.route_ranges(new_route_b[types], types)
            choices_a = new_route_a[types][ranges_a[0]:ranges_a[1]-1]
            if len(choices_a) > 2:
                # select two consecutive customers from route_a
                node_a1 = choice(choices_a)
                index_a1 = new_route_a[types].index(node_a1)
                index_a2 = index_a1 + 1
                node_a2 = new_route_a[types][index_a2]
                # validate swap with demand constraint
                nodes_demand_a = self.nodes[node_a1].get('demand') + self.nodes[node_a2].get('demand')
                route_demand_b = self.route_demand(new_route_b[types]) + nodes_demand_a
                # swap two consecutive customers
                if route_demand_b <= self.capacity:
                    index_b = randint(ranges_b[0],ranges_b[1])
                    new_route_b[types].insert(index_b, node_a2)
                    new_route_b[types].insert(index_b, node_a1)
                    new_route_a[types].remove(node_a1)
                    new_route_a[types].remove(node_a2)
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def swap_single_pair(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        swap a customer to a couple of customes from route to another route
        """
        new_route_a = [x[:] for x in route_a]
        new_route_b = [y[:] for y in route_b]
        types, i = randint(0, 1), 0 # 0: linehaul, 1: backhaul
        while i < 2:
            ranges_a = self.route_ranges(new_route_a[types], types)
            ranges_b = self.route_ranges(new_route_b[types], types)
            choices_a = new_route_a[types][ranges_a[0]:ranges_a[1]-1]
            choices_b = new_route_b[types][ranges_b[0]:ranges_b[1]]
            if len(choices_a) > 2 and len(choices_b) > 1:
                # select two consecutive customers from route_a
                node_a1 = choice(choices_a)
                index_a1 = new_route_a[types].index(node_a1)
                index_a2 = index_a1 + 1
                node_a2 = new_route_a[types][index_a2]
                # select one customer from route_b
                node_b = choice(choices_b)
                index_b = new_route_b[types].index(node_b)
                # validate swap with demand constraint
                nodes_demand_a = self.nodes[node_a1].get('demand') + self.nodes[node_a2].get('demand')
                nodes_demand_b = self.nodes[node_b].get('demand')
                route_demand_a = self.route_demand(new_route_a[types]) + nodes_demand_b - nodes_demand_a
                route_demand_b = self.route_demand(new_route_b[types]) + nodes_demand_a - nodes_demand_b
                # swap two consecutive customers
                if route_demand_b <= self.capacity and route_demand_a <= self.capacity:
                    new_route_b[types].insert(index_b, node_a2)
                    new_route_b[types].insert(index_b, node_a1)
                    new_route_a[types].insert(index_a1, node_b)
                    new_route_b[types].remove(node_b)
                    new_route_a[types].remove(node_a1)
                    new_route_a[types].remove(node_a2)
                    break
            types = abs(types - 1)
            i += 1
        return new_route_a, new_route_b


    def swap_backhauls(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        interchange the backhauls of two routes
        """
        new_route_a, new_route_b = [x[:] for x in route_a], [y[:] for y in route_b]
        new_route_a[1], new_route_b[1] = new_route_b[1], new_route_a[1]
        return new_route_a, new_route_b


    def divide_route(self, route: List[list]) -> list:
        """
        Divide a route into two routes
        """
        first_route = [x[:] for x in route]
        second_route = []
        if len(first_route[0]) > 2 and len(first_route[1]) > 2:
            second_route = [[0, first_route[0].pop(1)], [first_route[1].pop(-2), 0]]
        return first_route, second_route


    def shift_continuos_pair(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        Create a new route by shifting two consecutive customers, one from backhauls and another to linehauls
        """
        new_route_a = [x[:] for x in route_a]
        new_route_b = [y[:] for y in route_b]
        # get the first and last nodes's demands of route_a
        if len(new_route_a[0]) > 2 and len(new_route_a[1]) > 1:
            node_demand_linehaul = self.nodes[new_route_a[0][-1]].get('demand')
            node_demand_backhaul = self.nodes[new_route_a[1][0]].get('demand')
            route_demand_linehaul = self.route_demand(new_route_b[0]) + node_demand_linehaul
            route_demand_backhaul = self.route_demand(new_route_b[1]) + node_demand_backhaul
            # validate swap with demand constraint
            if route_demand_linehaul <= self.capacity and route_demand_backhaul <= self.capacity:
                node_linehaul_a = new_route_a[0].pop(-1)
                node_backhaul_a = new_route_a[1].pop(0)
                new_route_b[0].insert(len(new_route_b[0]), node_linehaul_a)
                new_route_b[1].insert(0, node_backhaul_a)
            # return the new routes
        return new_route_a, new_route_b


    def joint_linehauls(self, route_a: List[int], route_b: List[int]) -> List[int]:
        """
        Create a new route by joining two linehauls when backhauls are blank
        """
        # concatenate two linehauls by proximity
        a_first, a_last = route_a[0][1], route_a[0][-1]
        b_first, b_last = route_b[0][1], route_b[0][-1]
        # validate which route must be go in first
        if self.distance[a_first][b_last] < self.distance[a_last][b_first]:
            linehaul = [0] + route_b[0][1:] + route_a[0][1:]
        else: 
            linehaul = [0] + route_a[0][1:] + route_b[0][1:]
        # get the complete route
        route = [linehaul, route_b[1]] if route_a[1] == [0] else [linehaul, route_a[1]]
        return route

