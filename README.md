# SNEAT (Simplified NEAT)

A simplified implementation of [Neuro-evolution of Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf), a novel technique for neuro-evolution developed by Kenneth O. Stanley. 

## Why? 

Another implementation of NEAT in Python already exists, which is thorough and beautiful work. However, as things stand, it faces some deeper issues and thus doesn't perform in accordance with 
the benchmarks provided in the original paper, and is no longer maintained. This solution has cut down to the bone in an attempt to simplify both usage and the codebase.

Full disclosure, this was made for the fun of it, and is playing loose with the rules. 

## How? 

In the simplest case, all you need to begin training a neural network for any given problem is a fitness function. 

1. Install the package

`pip install sneat`

2. Set up your fitness function

Your fitness function should take a genome and output a fitness score based on how well that genome solves the task. 
Here's a simple example, training a neural network that will output the sum of its inputs:

```
def fitness_function(genome):
    inputs = list(np.random.randint(5, size=2))
    target = sum(inputs)
    output = genome.activate(inputs)
    difference = (output - target) ** 2
    fitness = 1 / difference
    return fitness
```

3. Magic

There's a bunch of hyperparameters that should be configured for your given problem, but again we'll take a simple approach and just use the default hyperparameters along with the default reporter:

```
from sneat import evolve

def fitness_function(genome):
    ...

winner = evolve(fitness_function)
```

...now watch the generations unfold and enjoy!

## Configuration and complexities

This repository is a WIP. Over the next few weeks, I'll get around to implementing the finishing touches. 