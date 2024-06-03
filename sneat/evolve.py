from sneat.population import Population
from sneat.config import get_config
import numpy as np
import pickle as pkl
from tabulate import tabulate as tb
from tqdm import tqdm
from multiprocessing import Pool

def save_genome(genome, filename):
    with open(filename, 'wb') as f:
        pkl.dump(genome, f)

def evaluate_genome(args):
    g, ff = args
    fitness = ff(g)
    return fitness

def evaluate_population(pop, ff):
    print('\n')
    with Pool() as p:
        # set the fitness of all genomes using evaluate_genome (which returns a single fitness score), and tqdm for multiprocessing
        fitness_scores = list(tqdm(p.imap(evaluate_genome, [(g, ff) for g in pop.genomes]), total=len(pop.genomes), desc='[-] Evaluating', leave=False))
        for g, fitness in zip(pop.genomes, fitness_scores):
            g.fitness = fitness

def save_checkpoint(pop):
    with open('checkpoint.pkl', 'wb') as f:
        pkl.dump(pop, f)

def load_checkpoint():
    try:
        with open('checkpoint.pkl', 'rb') as f:
            pop = pkl.load(f)
            print(f'[i] Restoring from checkpoint (gen. {pop.generation})...')
            return pop
    except FileNotFoundError:
        return None

def print_stats(pop):
    print(f'\n\n[i] Gen. {pop.generation}:')
    headers = ['Species', 'Members', 'Best Fitness', 'Average Fitness']
    for s in pop.species:
        s.members = sorted(s.members, key=lambda x: x.fitness, reverse=True)
    species = sorted(pop.species, key=lambda x: x.members[0].fitness, reverse=True)
    data = [[s.id, len(s.members), round(max(g.fitness for g in s.members), 2), round(np.mean([g.fitness for g in s.members]), 2)] for s in species]
    print(tb(data, headers=headers))
    print('-' * 55)

def evolve(fitness_function):
    config = get_config()
    
    pop = load_checkpoint() or Population()

    max_generations = config.getint('Evolution', 'max_generations') or np.inf
    max_fitness = config.getfloat('Evolution', 'max_fitness') or np.inf

    try:
        for _ in range(max_generations):
            
            # evaluate population
            evaluate_population(pop, fitness_function)

            # print stats
            print_stats(pop)

            # reproduce
            pop.reproduce()

            # save checkpoint
            if pop.generation % 10 == 0:
                save_checkpoint(pop)

            best_fitness = max(g.fitness for g in pop.genomes)
            if best_fitness >= max_fitness:
                winner = max(pop.genomes, key=lambda x: x.fitness)
                save_genome(winner, 'winner.pkl')
                print(f'\n\n[+] Winner found with fitness: {winner.fitness}\n\n')
                return winner
            
            if pop.generation >= max_generations:
                winner = max(pop.genomes, key=lambda x: x.fitness)
                save_genome(winner, 'winner.pkl')
                print(f'\n\n[+] Reached max generations, and achieved a fitness of: {winner.fitness}\n\n')
                return winner
                
    except KeyboardInterrupt:
            winner = max(pop.genomes, key=lambda x: x.fitness)
            print(f'\n\n[+] Best genome saved, with a fitness of {winner.fitness}\n')
            save_genome(winner, 'winner.pkl')
            return winner


    
