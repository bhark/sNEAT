import gymnasium as gym
import sys
import os
import numpy as np
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sneat import evolve

def fitness(genome, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else 'rgb_array')

    fitness = 1000
    obs, info = env.reset()
    for _ in range(fitness):
        obs = list(obs)
        action = genome.activate(obs)
        action = np.argmax(action)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            fitness -= 1 # penalize for falling

        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    return fitness

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            winner = pickle.load(f)
            fitness(winner, render=True)
    else:
        winner = evolve(fitness)
        fitness(winner, render=True)

if __name__ == '__main__':
    main()