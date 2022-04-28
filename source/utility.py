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
        try:
            for line in str(text).split('\n'):
                if ':' not in line:
                    items = line.split()
                    if len(items) == 2:
                        key = int(items[0]) - 1
                        if nodes.get(key):
                            nodes[key]['demand'] = int(items[1])
                        else:
                            nodes[key] = {'demand': int(items[1])}
                    elif len(items) == 4:
                        key = int(items[0]) - 1
                        if nodes.get(key):
                            nodes[key]['coordinates'] = (int(items[1]), int(items[2]))
                            nodes[key]['type_node'] = items[3]
                        else:
                            nodes[key] = {'coordinates': (int(items[1]), int(items[2])), 'type_node': items[3]}
                else:
                    key, value = line.split(':')
                    if key.strip() == 'CAPACITY':
                        capacity = int(value.strip())
            return {'nodes': nodes, 'capacity': capacity}

        except Exception as e:
            print('Error: invalid input file', e)
            exit(1)

