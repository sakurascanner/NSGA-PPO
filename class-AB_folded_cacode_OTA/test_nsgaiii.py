import array
import copy

import random

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import numpy as np
from deap import algorithms, base, benchmarks, creator, tools
from deap.tools.indicator import hv
from env_for_nsga_and_rl import AMPNMCF

from nsgaiii.selection import (
    associate,
    construct_hyperplane,
    find_extreme_points,
    find_ideal_point,
    generate_reference_points,
    normalize_objectives,
    sel_nsga_iii,
)

creator.create("FitnessMin3", base.Fitness, weights=(-1.0,) * 3)
creator.create("Individual3", array.array, typecode="d", fitness=creator.FitnessMin3)

def prepare_toolbox(
    problem_instance, selection_func, number_of_variables, bounds_low, bounds_up
):
    def uniform(low, up, size=None):
        try:
            return [random.uniform(a, b) for a, b in zip(low, up)]
        except TypeError:
            return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

    toolbox = base.Toolbox()

    toolbox.register("evaluate", problem_instance)
    toolbox.register("select", selection_func)

    toolbox.register("attr_float", uniform, bounds_low, bounds_up, number_of_variables)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual3, toolbox.attr_float
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "mate", tools.cxSimulatedBinaryBounded, low=bounds_low, up=bounds_up, eta=20.0
    )
    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=bounds_low,
        up=bounds_up,
        eta=20.0,
        indpb=1.0 / number_of_variables,
    )

    toolbox.pop_size = 42  # population size
    toolbox.max_gen = 200  # max number of iterations
    toolbox.mut_prob = 1 / number_of_variables
    toolbox.cross_prob = 0.29

    return toolbox

number_of_variables = 30

bounds_low, bounds_up = 0, 1

toolbox = prepare_toolbox(
    lambda ind: benchmarks.dtlz2(ind, 3),
    sel_nsga_iii,
    number_of_variables,
    bounds_low,
    bounds_up,
)

pop = toolbox.population(n=42)

for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

# ideal point (red star)
ideal_point = find_ideal_point(pop)
# extreme points marked (red)
extremes = find_extreme_points(pop)
# intercepts (in green)
intercepts = construct_hyperplane(pop, extremes)
verts = [(intercepts[0], 0, 0), (0, intercepts[1], 0), (0, 0, intercepts[2])]
# normalized objectives (light blue)
normalize_objectives(pop, intercepts, ideal_point)
# reference points (gray)
rps = generate_reference_points(3)
