import math
from matplotlib import pyplot as plt

class Graph():

    def __init__(self, nodes, capacity) -> None:
        """ 
        initializes a graph object
        depot is the first node -> 0
        """
        self.graph = {}
        self.capacity = capacity
        # add nodes to graph
        for key, value in dict(nodes).items():
            node_type = 'D' if key == '1' else 'L' if int(value['demand']) >= 0 else 'B'
            key = int(key) - 1
            self.graph[key] = {
                'demand': abs(int(value['demand'])),
                'coordinates': value['coordinates'],
                'type': node_type
            }


    def get_depot(self) -> dict:
        """
        returns the depot node
        """
        depot = {}
        for value in self.graph.values():
            if value['type'] == 'D':
                depot = value
                break
        return depot


    def get_linehauls(self) -> list:
        """
        returns a list of linehaul nodes
        """
        linehauls = []
        for value in self.graph.values():
            if value['type'] == 'L':
                linehauls.append(value)
        return linehauls


    def get_backhauls(self) -> list:
        """
        returns a list of backhaul nodes
        """
        backhauls = []
        for value in self.graph.values():
            if value['type'] == 'B':
                backhauls.append(value)
        return backhauls


    def get_distance(self, node1, node2) -> int:
        """
        returns the distance between two nodes
        """
        # get coordinates
        x1, y1 = self.graph[node1]['coordinates']
        x2, y2 = self.graph[node2]['coordinates']
        # calculate distance
        distance = math.sqrt((int(x2) - int(x1))**2 + (int(y2) - int(y1))**2)
        return distance #round(distance)


    def get_distance_route(self, route) -> int:
        """
        returns the distance of a route
        """
        distance = 0
        for i in range(len(route) - 1):
            distance += self.get_distance(route[i], route[i+1])
        return distance


    def polar_coordinates(self, final_node, initial_node = 0) -> tuple:
        """
        returns the distance and angle between depot and selected node with demand and type of node
        """
        # get coordinates
        x1, y1 = self.graph[initial_node]['coordinates'] # Depot
        x2, y2 = self.graph[final_node]['coordinates'] # Selected node
        # get demand and type of node
        demand = self.graph[final_node]['demand']
        node_type = self.graph[final_node]['type']
        # calculate distance and angle
        distance = math.sqrt((int(x2) - int(x1))**2 + (int(y2) - int(y1))**2)
        angle = math.degrees(math.atan2(int(y2) - int(y1), int(x2) - int(x1)))
        angle = angle if angle >= 0 else 360 + angle
        return (distance, angle, demand, node_type)


    def calculate_graph(self) -> tuple:
        """
        Calculate lists of x and y coordinates, colors, and node names
        """
        x, y, n, c = [], [], [], []
        for key, value in self.graph.items():
            x.append(int(value['coordinates'][0]))
            y.append(int(value['coordinates'][1]))
            n.append(key)
            c.append('c' if value['type'] == 'L' else 'r' if value['type'] == 'B' else 'g')
        return (x, y, n, c)


    def plot_graph(self, routes=[[1,2,3]]) -> None:
        """
        plots the graph without solution
        """
        x, y, n, c = self.calculate_graph()
        fig, ax = plt.subplots()
        ax.scatter(x, y, marker='o', c=c, edgecolor='b') # plot nodes
        # plot annotations
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
        # plot routes
        if routes:
            for route in routes:
                x_route, y_route = [], []
                for item in route:
                    x_route.append(int(self.graph[item]['coordinates'][0]))
                    y_route.append(int(self.graph[item]['coordinates'][1]))
                plt.plot(x_route, y_route, 'b')
        plt.show() # print the plot

    # Initial solution
    def initial_solution(self):
        """
        Initial solution with a greedy algorithm
        """
        linehauls, backhauls = [], []
        # Get linehauls and backhauls in polar coordinates
        for i in range(1, len(self.graph)):
            distance, angle, demand, type = self.polar_coordinates(i)
            if type == 'L':
                linehauls.append([i, distance, angle, demand])
            elif type == 'B':
                backhauls.append([i, distance, angle, demand])
        # Order linehauls and backhauls by angle
        sorted_linehauls = sorted(linehauls, key=lambda x: x[2])
        sorted_backhauls = sorted(backhauls, key=lambda x: x[2])
        # Iterate through linehauls and backhauls to create routes
        linehauls_roures = self.create_routes(sorted_linehauls)
        backhauls_routes = self.create_routes(sorted_backhauls)
        # Merge routes
        routes = linehauls_roures + backhauls_routes
        print(linehauls_roures, backhauls_routes)
        self.plot_graph(routes)


    def create_routes(self, sorted_items):
        """
        Create routes from sorted items with capacity restrictions
        """
        total_distance, route, routes = 0, [], []
        for item in sorted_items:
            if total_distance + item[3] <= self.capacity:
                route.append(item[0])
                total_distance += item[3]
            else:
                routes.append(route)
                total_distance, route = 0, []
                route.append(item[0])
                total_distance += item[3]
        if total_distance > 0:
            routes.append(route)
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

