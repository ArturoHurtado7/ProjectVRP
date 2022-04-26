#Variable neighborhood search

# -> sweep first, assignment second Aproach

# Initially two sets of open-ended routes are constructed 
# by sweeping through LH and BH nodes separately.

# A distance/cost matrix for the assignment problem is created 
# by including the distances between the end-nodes of the open-ended routes.

# A dummy route containing the depot is also added to the matrix 
# where a number of LH and BH routes are not equal.

# Data source:
# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/vrp/

from utility import Utility
from graph import Graph

def main():
    utility = Utility()
    input = utility.read_input('./data/eil22_50.vrp')
    values = utility.process_input(input)
    vpr = Graph(values.get('nodes'), values.get('capacity'))

    vpr.initial_solution()
    

    #print(vpr.get_depot())
    #print(vpr.get_backhauls())
    #print(vpr.get_linehauls())
    """for i in range(0, len(vpr.graph)):
        for j in range(0, len(vpr.graph)):
            #if i != j and i <= 11 and j > 11:
                distance = vpr.get_distance(vpr.graph[i], vpr.graph[j])
                #if distance in [17,69,22,72,9,49,70,30,42]:
                #if distance in [9]:
                print(i, j, distance)"""
    

    """vpr.plot_graph()
    node1 = 14
    node2 = 12
    distance1 = vpr.get_distance(node1, node2)
    print(node1, node2, distance1)
    node1 = 1
    node2 = 0
    distance2 = vpr.get_distance(vpr.graph[node1], vpr.graph[node2])
    print(node1, node2, distance2)
    node1 = 6
    node2 = 0
    distance3 = vpr.get_distance(vpr.graph[node1], vpr.graph[node2])
    print(node1, node2, distance3)
    print(distance1 + distance2 + distance3)"""

    #print(vpr.get_distance2((1,3), (1,1)))


    """sol1 = [0,7,10,11,9,19,18,21,20,0]
    route1 = vpr.get_distance_route(sol1)
    print(route1)
    sol2 = [0,8,5,4,3,1,17,12,16,14,15,13,0]
    route2 = vpr.get_distance_route(sol2)
    print(route2)
    sol3 = [0,2,6,0]
    route3 = vpr.get_distance_route(sol3)
    print(route3)
    print(route1 + route2 + route3)"""


if __name__ == '__main__':
    main()