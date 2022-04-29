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


if __name__ == '__main__':
    main()