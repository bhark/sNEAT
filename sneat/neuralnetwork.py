import numpy as np
from .config import activation_functions
from .normalizer import Normalizer
import networkx as nx
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, callbacks):
        if callbacks['config'].getboolean('NeuralNetwork', 'use_normalizer'):
            self.normalizer = Normalizer(callbacks['config'].getint('NeuralNetwork', 'num_inputs'))
        self.node_counter = 0
        self.nodes, self.connections = [], []
        self.callbacks = callbacks # this should contain the "find_or_create_innovation" method
        self.initialize(callbacks['config'].getint('NeuralNetwork', 'num_inputs'), callbacks['config'].getint('NeuralNetwork', 'num_outputs'))

    def next_node_id(self):
        self.node_counter += 1
        return self.node_counter

    def initialize(self, num_input, num_output):
        self.nodes = [Node(self.next_node_id(), node_type='input', callbacks=self.callbacks) for _ in range(num_input)]
        self.nodes += [Node(self.next_node_id(), node_type='output', callbacks=self.callbacks) for _ in range(num_output)]
        
        # create one random connection
        in_node = np.random.choice([n for n in self.nodes if n.node_type == 'input'])
        out_node = np.random.choice([n for n in self.nodes if n.node_type == 'output'])
        innovation_number = self.callbacks['find_or_create_innovation'](in_node, out_node)
        self.add_connection(in_node, out_node)

    def feed_forward(self, inputs):
        # check if inputs match the number of input nodes
        input_nodes = [n for n in self.nodes if n.node_type == 'input']
        if len(inputs) != len(input_nodes):
            raise ValueError(f'Wrong input shape. Expected {len(input_nodes)} inputs, but got {len(inputs)}.')

        if hasattr(self, 'normalizer'):
            self.normalizer.observe(inputs)
            inputs = self.normalizer.normalize(inputs)

        # convert inputs to np array
        inputs = np.array(inputs)

        # reset node values
        for node in self.nodes:
            node.value = 0.0
            node.ready = False

        node_dict = {node.id: node for node in self.nodes}

        # assign input values
        for node, input_value in zip(input_nodes, inputs):
            node_dict[node.id].value = input_value
            node_dict[node.id].ready = True

        # assign hidden and output values
        while not all(node.ready for node in node_dict.values()):
            for node in node_dict.values():
                if node.ready:
                    continue

                # get the incoming connections
                incoming_connections = [c for c in self.connections if c.out_node == node and c.enabled]

                if not incoming_connections:
                    node.ready = True
                    node.value = activation_functions[node.activation](node.value)
                    continue

                # check if all incoming connections are ready
                if all(node_dict[c.in_node.id].ready for c in incoming_connections):
                    # calc node value
                    node_dict[node.id].value = np.sum([node_dict[c.in_node.id].value * c.weight for c in incoming_connections]) + node_dict[node.id].bias
                    node_dict[node.id].value = activation_functions[node.activation](node_dict[node.id].value)
                    node.ready = True

        # return output values
        output_nodes = [node for node in self.nodes if node.node_type == 'output']
        return [node_dict[node.id].value for node in output_nodes]

    def add_random_node(self):
        '''
        inserts a new node onto a random connection,
        splitting it in two
        '''

        try:
            connection = np.random.choice([c for c in self.connections if c.enabled])
        except ValueError:
            return

        new_node = Node(self.next_node_id(), node_type='hidden', callbacks=self.callbacks)

        # create two new connections
        in_node = connection.in_node
        out_node = connection.out_node
        innovation_number1 = self.callbacks['find_or_create_innovation'](in_node, new_node)
        innovation_number2 = self.callbacks['find_or_create_innovation'](new_node, out_node)
        try:
            self.nodes.append(new_node)
            self.add_connection(in_node, new_node)
            self.add_connection(new_node, out_node)
            connection.enabled = False
        except ValueError:
            return

    def add_random_connection(self):
        '''
        creates a random connection between two nodes
        '''

        retries = 0
        max_retries = 10

        while retries < max_retries:
            in_node = np.random.choice([n for n in self.nodes if n.node_type != 'output'])
            out_node = np.random.choice([n for n in self.nodes if n.node_type != 'input'])
            try:
                self.add_connection(in_node, out_node)
                break
            except ValueError:
                retries += 1

    def remove_random_node(self):
        '''
        removes a random hidden node
        '''

        hidden_nodes = [n for n in self.nodes if n.node_type == 'hidden']
        if not hidden_nodes:
            return

        node = np.random.choice(hidden_nodes)
        conns = [c for c in self.connections if c.in_node == node or c.out_node == node]
        for conn in conns:
            self.connections.remove(conn)
        self.nodes.remove(node)

    def change_random_weight(self):
        '''
        changes the weight of a random connection
        '''

        connections = [c for c in self.connections if c.enabled]
        if not connections:
            return

        connection = np.random.choice(connections)
        connection.weight += np.random.normal(-0.1, 0.1)

    def change_random_bias(self):
        '''
        changes the bias of a random node
        '''

        nodes = [n for n in self.nodes if n.node_type != 'input']
        if not nodes:
            return

        node = np.random.choice(nodes)
        node.bias += np.random.normal(-0.1, 0.1)

    def change_random_activation(self):
        '''
        changes the activation function of a random node
        '''

        nodes = [n for n in self.nodes if n.node_type != 'input']
        if not nodes:
            return

        node = np.random.choice(nodes)
        node.activation = np.random.choice(list(activation_functions.keys()))

    def toggle_random_connection(self):
        '''
        toggles the enabled state of a random connection
        '''

        connections = [c for c in self.connections if c.enabled]
        if not connections:
            return

        connection = np.random.choice(connections)
        connection.enabled = not connection.enabled

    def add_connection(self, in_node, out_node):
        # check if connection already exists
        if any(c.in_node == in_node and c.out_node == out_node for c in self.connections):
            raise ValueError('Connection already exists')

        # check if connection creates cycles
        if self.would_create_cycle(in_node, out_node):
            raise ValueError('Connection would create a cycle')

        # create the connection
        innovation_number = self.callbacks['find_or_create_innovation'](in_node, out_node)
        self.connections.append(Connection(innovation_number, in_node, out_node))

    def would_create_cycle(self, in_node, out_node):
        '''
        checks if adding a connection between two nodes would create a cycle
        '''

        # create a graph from the connections
        G = nx.DiGraph()
        for connection in self.connections:
            if connection.enabled:
                G.add_edge(connection.in_node.id, connection.out_node.id)

        # add the new connection
        G.add_edge(in_node.id, out_node.id)

        # check if the graph has a cycle
        try:
            nx.find_cycle(G)
            return True
        except nx.exception.NetworkXNoCycle:
            return False

    def visualize(self):
        G = nx.DiGraph()

        # Add nodes to the graph
        for node in self.nodes:
            if node.node_type == 'input':
                G.add_node(node.id, pos=(0, node.id))
            elif node.node_type == 'output':
                G.add_node(node.id, pos=(1, node.id))
            else:
                # Generate random position for hidden nodes
                x = rng.uniform(0.2, 0.8)
                y = rng.uniform(0, 3)
                G.add_node(node.id, pos=(x, y))

        # Add edges to the graph
        for connection in self.connections:
            if connection.enabled:
                G.add_edge(connection.in_node.id, connection.out_node.id, weight=connection.weight)

        # Get the positions of the nodes
        pos = nx.get_node_attributes(G, 'pos')

        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

        # Draw the edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)

        # Draw the edge labels (connection weights)
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

        # Draw the node labels (node IDs)
        node_labels = {node.id: str(node.id) for node in self.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels)

        plt.axis('off')
        plt.show()

class Node:
    def __init__(self, id=None, node_type='hidden', activation=None, callbacks=None):
        if not id:
            raise ValueError('Node must have an id')

        self.id = id
        self.node_type = node_type
        self.bias = np.random.uniform(-1, 1)
        self.callbacks = callbacks
        self.activation = self.initialize_activation()
        self.value = 0.0
        self.ready = False

    def initialize_activation(self):
        type_to_activation = {
            'input': self.callbacks['config'].get('NeuralNetwork', 'input_activation'),
            'output': self.callbacks['config'].get('NeuralNetwork', 'output_activation'),
            'hidden': np.random.choice(list(activation_functions.keys())) }

        # get the name of the activation function we'll use
        activation_name = type_to_activation[self.node_type]
        return activation_name

class Connection:
    def __init__(self, innovation_number, in_node, out_node, enabled=True):
        self.in_node = in_node
        self.out_node = out_node
        self.innovation_number = innovation_number
        self.weight = np.random.uniform(-1, 1)
        self.enabled = enabled      

        if self.in_node.node_type == 'output':
            raise ValueError('Output nodes cannot have outgoing connections')
        if self.out_node.node_type == 'input':
            raise ValueError('Input nodes cannot have incoming connections')
        if self.in_node == self.out_node:
            raise ValueError('Connection cannot be made between the same node')

