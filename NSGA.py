import copy
import random
import csv
import numpy as np
from deap import tools,creator,base
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import deap
from selection import sel_nsga_iii


class NSGA_Agent:

    def __init__(self, population_size=20, generation=10):
        self.generation = generation
        creator.create("FitnessMin", base.Fitness, weights=((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.population = [creator.Individual(self.generate_individual()) for _ in range(population_size)]
        self.param_range_min = [1,1,0,
                   1,1,0,
                   1,1,0,
                   1,1,100,
                   1,1,0,
                   1,1,0,
                   1,1,0,
                   0,5,5]

        self.param_range_max = [5,5,20,
                   5,5,20,
                   5,5,20,
                   5,5,500,
                   5,5,20,
                   5,5,20,
                   5,5,20,
                   1e-5,15,15]
        
        self.param_int = [0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,1,1,]


    def generate_individual(self):
        num_triplets = 7
        individual = []
        
        for _ in range(num_triplets):
            individual.extend([random.uniform(1, 5), random.uniform(1, 5), random.randint(0, 20)])
        individual.extend([random.uniform(0, 1e-5), random.randint(5, 15), random.randint(5, 15)])
        individual[11]=random.randint(100,500)
        return individual
    
    def variation(self, individual, mutation_prob=0.1, eta=20):
        mutated = list(copy.deepcopy(individual))
        for i in range(len(mutated)):
            if random.random() < mutation_prob:
                x = mutated[i]
                min_val = self.param_range_min[i]
                max_val = self.param_range_max[i]
                delta = min(x - min_val, max_val - x)
                u = random.random()
                if u <= 0.5:
                    delta_q = (2*u)**(1/(eta+1)) - 1
                else:
                    delta_q = 1 - (2*(1-u))**(1/(eta+1))
                mutated[i] = x + delta_q * delta
                mutated[i] = max(min_val, min(mutated[i], max_val))
                if self.param_int[i] ==  1:
                    mutated[i] = int(round(mutated[i]))
        return creator.Individual(mutated)
    
    def crossover(self, parent1, parent2, eta=15):
        child1 = []
        child2 = []
        for i, (p1, p2) in enumerate(zip(parent1, parent2)):
            min_val = self.param_range_min[i]
            max_val = self.param_range_max[i]
            if random.random() < 0.5:
                u = random.random()
                beta = (2*u)**(1/(eta+1)) if u < 0.5 else (1/(2*(1-u)))**(1/(eta+1))
                c1 = 0.5*((1+beta)*p1 + (1-beta)*p2)
                c2 = 0.5*((1-beta)*p1 + (1+beta)*p2)
                c1 = max(min_val, min(c1, max_val))
                c2 = max(min_val, min(c2, max_val))
                child1.append(c1)
                child2.append(c2)
            else:
                child1.append(p1)
                child2.append(p2)
        return creator.Individual(child1), creator.Individual(child2)
    
    def crossover_variation(self, individuals):
        offspring = []
        for i in range(0, len(individuals)-1, 2):
            parent1, parent2 = individuals[i], individuals[i+1]
            child1, child2 = self.crossover(parent1, parent2)
            offspring.append(self.variation(child1))
            offspring.append(self.variation(child2))
        return offspring
    
    def add_individual(self, parameters, info):
        evaluated_value = (info['Power'], info['dcgain'], info['GBW'], info['phase_margin (deg)'], info['TC'], info['vos'], info['cmrrdc'], info['PSRP'], info['PSRN'], info['sr'], info['setting_time'])
        ind = creator.Individual(parameters)
        ind.fitness.values = evaluated_value
        return ind
    
    #def selection(self, selection):
    #    selected_individuals = sel_nsga_iii(self.population, k=selection)
    #    return [np.array(ind) for ind in selected_individuals]