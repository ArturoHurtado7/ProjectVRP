import math

class Utility:

    def read_input(self, input_path):
        """
        Read the input file
        """
        try:
            with open(input_path, 'r', encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f'Error: file "{input_path}" not found')
            exit(1)


    def process_input(self, text):
        """
        Process the input file and return an info dictionary
        """
        nodes = {}
        capacity = 0
        for line in str(text).split('\n'):
            if ':' not in line:
                items = line.split()
                if len(items) == 2:
                    if nodes.get(items[0]):
                        nodes[items[0]]['demand'] = int(items[1])
                    else:
                        nodes[items[0]] = {'demand': int(items[1])}
                elif len(items) == 3:
                    if nodes.get(items[0]):
                        nodes[items[0]]['coordinates'] = (int(items[1]), int(items[2]))
                    else:
                        nodes[items[0]] = {'coordinates': (int(items[1]), int(items[2]))}
            else:
                key, value = line.split(':')
                if key.strip() == 'CAPACITY':
                    capacity = int(value.strip())
        return {'nodes': nodes, 'capacity': capacity}