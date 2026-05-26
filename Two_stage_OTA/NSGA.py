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

    def __init__(self, population_size=384):
        creator.create("FitnessMin", base.Fitness, weights=((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.population = [creator.Individual(self.generate_individual()) for _ in range(population_size)]
        self.delta_action = np.array([
            0.5,    # W_diff
            0.1,    # L_diff (³¤¶Èµ÷½ÚÐèÒª¸ü¾«Ï¸)
            0.5,    # W_load1
            0.1,    # L_load1
            0.5,    # W_pmos_mirror
            0.1,    # L_pmos_mirror
            1.0,    # W_gm2 (´ó³ß´ç¹Ü£¬²½³¤¿ÉÒÔ´óÒ»µã)
            0.01,   # L_gm2 (¶Ì¹µµÀ£¬ÐèÒª¼«¸ß¾«¶È£¬Èç 0.15 -> 0.16)
            1,      # M_out_stage (±ØÐëÊÇÕûÊý)
            0.5,    # W_bias_n
            0.5,    # L_bias_n
            0.1,    # W_Rc (µç×èµ÷½ÚÁéÃô)
            1,# Cc_val (0.1pF)
            0.5  # current_0_bias (0.5uA)
        ])

        self.param_range_min = np.array([
            2.0,    # W_diff: ÊäÈë¶Ô²»ÄÜÌ«Ð¡£¬·ñÔòÔëÉùºÍÊ§Åä´ó
            0.5,    # L_diff
            1.0,    # W_load1
            0.5,    # L_load1
            1.0,    # W_pmos_mirror
            0.5,    # L_pmos_mirror
            5.0,    # W_gm2: ±ØÐë×ã¹»´óÒÔÌá¹© Gm
            0.15,   # L_gm2: ÔÊÐíÊ¹ÓÃ¹¤ÒÕ×îÐ¡Öµ (150nm) »»È¡ËÙ¶È
            2,      # M_out_stage: ÖÁÉÙÊÇ»ù×¼µçÁ÷µÄ2±¶
            0.5,    # W_bias_n
            1.0,    # L_bias_n: Æ«ÖÃµçÂ·½¨Òé³¤Ò»µã£¬Ôö¼ÓÊä³ö×è¿¹
            0.5,    # W_Rc
            1,# Cc_val: ×îÐ¡ 0.1pF
            1.0  # current_0_bias: ×îÐ¡ 1uA
        ])

        self.param_range_max = np.array([
            100.0,  # W_diff
            5.0,    # L_diff
            50.0,   # W_load1
            5.0,    # L_load1
            50.0,   # W_pmos_mirror
            5.0,    # L_pmos_mirror
            200.0,  # W_gm2: Êä³ö¼¶Í¨³£ºÜ´ó
            1.0,    # L_gm2
            100,    # M_out_stage: ×î´ó100±¶µçÁ÷
            20.0,   # W_bias_n
            10.0,   # L_bias_n
            10.0,   # W_Rc
            50,# Cc_val: ×î´ó 10pF (Æ¬ÉÏµçÈÝºÜÕ¼Ãæ»ý)
            50.0 # current_0_bias: ×î´ó 50uA
        ])


    def generate_individual(self):
        individual = []
        individual.extend([random.uniform(2,100), random.uniform(0.5, 5)])
        individual.extend([random.uniform(1,50), random.uniform(0.5, 5)])
        individual.extend([random.uniform(1,50), random.uniform(0.5, 5)])
        individual.extend([random.uniform(5,50), random.uniform(0.15, 5)])
        individual.extend([random.randint(2,100)])
        individual.extend([random.uniform(0.5,20),random.uniform(1,10)])
        individual.extend([random.uniform(0.5,10)])
        individual.extend([random.uniform(0.1,20)])
        individual.extend([random.uniform(1,50)])
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
                #if self.param_int[i] ==  1:
                #    mutated[i] = int(round(mutated[i]))
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
        evaluated_value = (info['Power'][1], info['dcgain'][1], info['GBW'][1], info['phase_margin (deg)'][1], info['TC'][1], info['vos'][1], info['cmrrdc'][1], info['PSRP'][1], info['PSRN'][1], info['sr'][1], info['settlingTime'][1], info['reward'])
        print("individual_eval_val :", evaluated_value)
        ind = creator.Individual(parameters)
        ind.fitness.values = evaluated_value
        return ind
    
    #def selection(self, selection):
    #    selected_individuals = sel_nsga_iii(self.population, k=selection)
    #    return [np.array(ind) for ind in selected_individuals]