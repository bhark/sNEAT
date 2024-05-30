# sNEAT (Simplified NEAT)

![GitHub Tag](https://img.shields.io/github/v/tag/bhark/sneat) 
![PyPI - Version](https://img.shields.io/pypi/v/sneat)
![PyPI - License](https://img.shields.io/pypi/l/sneat)

A simplified implementation of [Neuro-evolution of Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), a novel technique for neuro-evolution developed by Kenneth O. Stanley. 

## Why? 

Another implementation of NEAT in Python already exists (by CodeReclaimers, [here](https://github.com/CodeReclaimers/neat-python)), which is thorough and really nice work. However, as things stand, it faces some deeper issues and thus doesn't perform in accordance with the benchmarks provided in the original paper, and is no longer maintained. This solution has cut down to the bone in an attempt to simplify both usage and the codebase, and to achieve the expected results. 

Full disclosure, this was made for the fun of it, and is playing fast and loose with the rules. 

## How? 

In the simplest case, all you need to begin training a neural network for any given problem is a fitness function. 

1. Install the package

`$ pip install sneat`

2. Set up your fitness function

Your fitness function should take a genome and output a fitness score based on how well that genome solves the task. 
Here's a simple example, training a neural network that will output the sum of its inputs:

```
def fitness_function(genome):
    inputs = list(np.random.randint(5, size=2))
    target = sum(inputs)

    # feed input to the genomes neural network, return its output
    output = genome.activate(inputs) 

    difference = (output - target) ** 2
    fitness = 1 / difference
    return fitness
```

3. Magic

There's a bunch of hyperparameters that can (should) be configured for your given problem, but again we'll take a simple approach and just use the default hyperparameters along with the default evolution loop:

```
from sneat import evolve

def fitness_function(genome):
    ...

winner = evolve(fitness_function)
```

...now watch the generations unfold and enjoy! If you let it run long enough, we might get to experience our very own doomsday. The `evolve` function outputs the winning genome when one of the following three conditions are fulfilled: 

- The fitness threshold is reached by the best genome in the population
- The maximum number of generations are reached
- The user cancels (`CTRL+C`)

## Configuration and complexities

A default configuration file is supplied, but you'll probably want to change some details (such as the number of input and output nodes in the networks). You can include as few or as many configuration elements as you want; those you don't provide will fall back to the defaults. 

Create a `config.ini` file in your working directory with the settings you want to change. Here's the default config file for inspiration: 

```
[NeuralNetwork]
num_inputs = 2
num_outputs = 1
input_activation = linear
output_activation = sigmoid
use_normalizer = False

[Population]
population_size = 150
compatibility_threshold = 3.0
min_species_size = 5
elite_size = 3
survival_threshold = 0.2

[MutationRates]
add_node=0.1
add_connection=0.2
toggle_connection=0.08
change_weight=0.65
change_activation=0.05
change_bias=0.05
remove_node=0.08

[Evolution]
max_generations = 100 # set to 0 to disable
max_fitness = 4 # set to 0 to disable
```

## More control

If you want to have more control over the whole loop (for custom reporting, for example), I'd suggest importing the `Population` class and working around that. This class has `.reproduce()`, which will perform selection, cross-over and mutation on all genomes based on their fitness values. Finally, it will properly speciate the new genomes and move on to the next generation. 

`Population.species` is a list containing all the species, which in turn offers `Species.genomes`. I'll let you figure out the rest - the code is pretty straight-forward. 