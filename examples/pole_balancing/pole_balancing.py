import gymnasium as gym
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from sneat import evolve

def fitness(genome, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else 'rgb_array')

    fitnesses = []
    for _ in range(20):
        fitness = 0

        obs, info = env.reset()
        while True:
            obs = list(obs)
            action = genome.activate(obs)
            action = np.argmax(action)
            obs, reward, terminated, truncated, info = env.step(action)

            fitness += reward

            if terminated or truncated:
                break

        fitnesses.append(fitness)
    
    env.close()
    return np.mean(fitnesses)

def main():
    winner = evolve(fitness)
    fitness(winner, render=True)

if __name__ == '__main__':
    main()