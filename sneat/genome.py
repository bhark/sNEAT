from .neuralnetwork import NeuralNetwork
from .config import get_config
import numpy as np
from copy import deepcopy
import time
import pickle

class Genome:
    def __init__(self, callbacks):
        self.callbacks = callbacks
        self.fitness = 0
        self.normalized_fitness = 0
        self.adjusted_fitness = 0
        self.id = callbacks['get_next_genome_id']()
        self.network = NeuralNetwork(callbacks)

    @staticmethod
    def crossover(g1, g2):
        most_fit, least_fit = (g1, g2) if g1.fitness > g2.fitness else (g2, g1)
        child = most_fit.clone()

        # dicts for faster lookups
        least_fit_conns = {c.innovation_number: c for c in least_fit.network.connections}
        least_fit_nodes = {n.id: n for n in least_fit.network.nodes}

        # inherit weights randomly from either parent
        for c in child.network.connections:
            if c.innovation_number in least_fit_conns:
                c.weight = np.random.choice([c.weight, least_fit_conns[c.innovation_number].weight])

        # the same goes for biases and activation functions
        for n in child.network.nodes:
            if n.id in least_fit_nodes:
                n.bias = np.random.choice([n.bias, least_fit_nodes[n.id].bias])
                n.activation = least_fit_nodes[n.id].activation

        return child

    def mutate(self):
        mutation_rates = {
            'add_node': self.callbacks['config'].getfloat('MutationRates', 'add_node'),
            'add_connection': self.callbacks['config'].getfloat('MutationRates', 'add_connection'),
            'change_weight': self.callbacks['config'].getfloat('MutationRates', 'change_weight'),
            'change_activation': self.callbacks['config'].getfloat('MutationRates', 'change_activation'),
            'toggle_connection': self.callbacks['config'].getfloat('MutationRates', 'toggle_connection'),
            'change_bias': self.callbacks['config'].getfloat('MutationRates', 'change_bias'),
            'remove_node': self.callbacks['config'].getfloat('MutationRates', 'remove_node')
        }

        total = sum(mutation_rates.values())
        mutation_rates = {k: v / total for k, v in mutation_rates.items()}
        mutation = np.random.choice(list(mutation_rates.keys()), p=list(mutation_rates.values()))

        mutation_functions = {
            'add_node': self.network.add_random_node,
            'add_connection': self.network.add_random_connection,
            'change_weight': self.network.change_random_weight,
            'change_activation': self.network.change_random_activation,
            'toggle_connection': self.network.toggle_random_connection,
            'change_bias': self.network.change_random_bias,
            'remove_node': self.network.remove_random_node
        }
        
        mutation_functions[mutation]()

    def clone(self):
        copy = pickle.loads(pickle.dumps(self))
        return copy

    def activate(self, inputs):
        '''
        alias for network.feed_forward
        '''
        return self.network.feed_forward(inputs)