from sneat.population import Population
from sneat.config import get_config
import numpy as np
import pickle as pkl
from tabulate import tabulate as tb
from tqdm import tqdm
import multiprocessing as mp

def save_genome(genome, filename):
    with open(filename, 'wb') as f:
        pkl.dump(genome, f)

def evaluate_genome(args):
    g, ff = args
    fitness = ff(g)
    return fitness

def evaluate_population(pop, ff):
    print('\n')
    with mp.Pool(mp.cpu_count() - 1) as p:
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
            print(f'[+] Loaded checkpoint (gen. {pop.generation})')
            return pop
    except FileNotFoundError:
        return None

def print_stats(pop):

    # sort by fitness
    for s in pop.species:
        s.members = sorted(s.members, key=lambda x: x.fitness, reverse=True)
    species = sorted(pop.species, key=lambda x: x.members[0].fitness, reverse=True)

    # print general stats
    print(f'\n[i] Generation: {pop.generation}')
    print(f'[i] Compatibility threshold: {round(pop.compatibility_threshold, 2)}')
    print(f'[i] Population size: {len(pop.genomes)}')
    print(f'[i] Species: {len(pop.species)}')
    print(f'[i] Average fitness: {round(np.mean([g.fitness for g in pop.genomes]), 2)}')
    print(f'[i] Best fitness: {round(max(g.fitness for g in pop.genomes), 2)} (best ever: {round(pop.best_genome_seen.fitness, 2) if pop.best_genome_seen else 'N/A'})')
    print('-' * 85)

    # print species
    headers = ['Species', 'Members', 'Best Fitness', 'Average Fitness', 'Stagnation', 'Best Complexity']
    data = [[s.id, len(s.members), round(max(g.fitness for g in s.members), 2), round(np.mean([g.fitness for g in s.members]), 2), s.stagnation, f'{len(s.members[0].network.nodes)}n + {len(s.members[0].network.connections)}'] for s in species]
    print(tb(data, headers=headers))
    print('-' * 85)

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
            print(f'[-] Reproducing...', end='\r', flush=True)
            pop.reproduce()
            print(f'[+] Reproduced                                   ')

            # save checkpoint
            if pop.generation % 10 == 0:
                save_checkpoint(pop)

            best_fitness = max(g.fitness for g in pop.genomes)

            if pop.best_genome_seen is None or best_fitness > pop.best_genome_seen.fitness:
                new_best = max(pop.genomes, key=lambda x: x.fitness)
                print(f'\n\n[+] New best genome found with fitness: {round(new_best.fitness, 2)} (previous was {round(pop.best_genome_seen.fitness, 2) if pop.best_genome_seen else 'N/A'})')
                pop.best_genome_seen = new_best.clone()

            if best_fitness >= max_fitness:
                save_genome(best_genome_seen, 'winner.pkl')
                print(f'\n\n[+] Winner found with fitness: {pop.best_genome_seen.fitness}\n\n')
                return pop.best_genome_seen
            
            if pop.generation >= max_generations:
                save_genome(pop.best_genome_seen, 'winner.pkl')
                print(f'\n\n[+] Reached max generations, and achieved a fitness of: {pop.best_genome_seen.fitness}\n\n')
                return pop.best_genome_seen
                
    except KeyboardInterrupt:
            print(f'\n\n[+] Best genome saved, with a fitness of {pop.best_genome_seen.fitness}\n')
            save_genome(pop.best_genome_seen, 'winner.pkl')
            return pop.best_genome_seen


    
