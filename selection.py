#    This file is part of nsgaiii, a Python implementation of NSGA-III.
#
#    nsgaiii is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    nsgaiii is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with nsgaiii. If not, see <http://www.gnu.org/licenses/>.
#
#    by Luis Marti http://lmarti.com

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

class ReferencePoint(list):
    """A reference point exists in objective space an has a set of individuals
    associated to it."""

    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []


def generate_reference_points(num_objs, num_divisions_per_obj=4):
    """Generates reference points for NSGA-III selection. This code is based on
    jMetal NSGA-III implementation <https://github.com/jMetal/jMetal>.
    """

    def gen_refs_recursive(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left / total
            ref = ReferencePoint(copy.deepcopy(work_point))
            return [ref]
        else:
            res = []
            for i in range(left + 1):
                work_point[depth] = i / total
                res = res + gen_refs_recursive(
                    work_point, num_objs, left - i, total, depth + 1
                )
            return res

    return gen_refs_recursive(
        [0] * num_objs,
        num_objs,
        num_divisions_per_obj,#num_objs,# * 
        num_divisions_per_obj,#num_objs,# * 
        0,
    )


def find_ideal_point(individuals):
    "Finds the ideal point from a set individuals."
    current_ideal = [np.infty] * len(individuals[0].fitness.values)
    for ind in individuals:
        # Use wvalues to accomodate for maximization and minimization problems.
        current_ideal = np.minimum(current_ideal, np.multiply(ind.fitness.wvalues, -1))
    return current_ideal


def find_extreme_points(individuals):
    "Finds the individuals with extreme values for each objective function."
    return [
        sorted(individuals, key=lambda ind: ind.fitness.wvalues[o] * -1)[-1]
        for o in range(len(individuals[0].fitness.values))
    ]


def construct_hyperplane(individuals, extreme_points):
    "Calculates the axis intersects for a set of individuals and its extremes."

    def has_duplicate_individuals(individuals):
        for i in range(len(individuals)):
            for j in range(i + 1, len(individuals)):
                if individuals[i].fitness.values == individuals[j].fitness.values:
                    return True
        return False

    num_objs = len(individuals[0].fitness.values)

    if has_duplicate_individuals(extreme_points):
        intercepts = [extreme_points[m].fitness.values[m] for m in range(num_objs)]
    else:
        b = np.ones(num_objs)
        A = [point.fitness.values for point in extreme_points]
        x = np.linalg.solve(A, b)
        intercepts = 1 / x
    return intercepts


def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
    "Normalizes an objective."
    # Numeric trick present in JMetal implementation.
    if np.abs(intercepts[m] - ideal_point[m] > epsilon):
        return individual.fitness.values[m] / (intercepts[m] - ideal_point[m])
    else:
        return individual.fitness.values[m] / epsilon


def normalize_objectives(individuals, intercepts, ideal_point):
    """Normalizes individuals using the hyperplane defined by the intercepts as
    reference. Corresponds to Algorithm 2 of Deb & Jain (2014)."""
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        ind.fitness.normalized_values = list(
            [
                normalize_objective(ind, m, intercepts, ideal_point)
                for m in range(num_objs)
            ]
        )
    return individuals


def perpendicular_distance(direction, point):
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(
        np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2)
    )
    return np.sqrt(d)


def associate(individuals, reference_points):
    """Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014)."""
    tools.sortLogNondominated(individuals, len(individuals))
    len(individuals[0].fitness.values)

    for ind in individuals:
        rp_dists = [
            (rp, perpendicular_distance(ind.fitness.normalized_values, rp))
            for rp in reference_points
        ]
        best_rp, best_dist = sorted(rp_dists, key=lambda rpd: rpd[1])[0]
        ind.reference_point = best_rp
        ind.ref_point_distance = best_dist
        best_rp.associations_count += 1  # update de niche number
        best_rp.associations += [ind]


def niching_select(individuals, k):
    """Secondary niched selection based on reference points. Corresponds to
    steps 13-17 of Algorithm 1 and to Algorithm 4."""
    if len(individuals) == k:
        return individuals

    # individuals = copy.deepcopy(individuals)

    ideal_point = find_ideal_point(individuals)
    extremes = find_extreme_points(individuals)
    intercepts = construct_hyperplane(individuals, extremes)
    normalize_objectives(individuals, intercepts, ideal_point)

    reference_points = generate_reference_points(len(individuals[0].fitness.values))

    associate(individuals, reference_points)

    res = []
    while len(res) < k:
        min_assoc_rp = min(reference_points, key=lambda rp: rp.associations_count)
        min_assoc_rps = [
            rp
            for rp in reference_points
            if rp.associations_count == min_assoc_rp.associations_count
        ]
        chosen_rp = min_assoc_rps[random.randint(0, len(min_assoc_rps) - 1)]

        # print('Rps',min_assoc_rp.associations_count, chosen_rp.associations_count, len(min_assoc_rps))

        if chosen_rp.associations:
            if chosen_rp.associations_count == 0:
                sel = min(
                    chosen_rp.associations, key=lambda ind: ind.ref_point_distance
                )
            else:
                sel = chosen_rp.associations[
                    random.randint(0, len(chosen_rp.associations) - 1)
                ]
            res += [sel]
            chosen_rp.associations.remove(sel)
            chosen_rp.associations_count += 1
            individuals.remove(sel)
        else:
            reference_points.remove(chosen_rp)
    return res


def sel_nsga_iii(individuals, k):
    """Implements NSGA-III selection as described in
    Deb, K., & Jain, H. (2014). An Evolutionary Many-Objective Optimization
    Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
    Part I: Solving Problems With Box Constraints. IEEE Transactions on
    Evolutionary Computation, 18(4), 577-601. doi: 10.1109/TEVC.2013.2281535.
    """
    assert len(individuals) >= k

    if len(individuals) == k:
        return individuals

    # Algorithm 1 steps 4--8
    fronts = tools.sortLogNondominated(individuals, len(individuals))

    limit = 0
    res = []
    for f, front in enumerate(fronts):
        res += front
        if len(res) > k:
            limit = f
            break
    # Algorithm 1 steps
    selection = []
    if limit > 0:
        for f in range(limit):
            selection += fronts[f]

    # complete selected inividuals using the referece point based approach
    selection += niching_select(fronts[limit], k - len(selection))
    return selection


__all__ = ["sel_nsga_iii"]

param_range_min = [1,1,0,
                   1,1,0,
                   1,1,0,
                   1,1,100,
                   1,1,0,
                   1,1,0,
                   1,1,0,
                   0,5,5]

param_range_max = [5,5,20,
                   5,5,20,
                   5,5,20,
                   5,5,500,
                   5,5,20,
                   5,5,20,
                   5,5,20,
                   1e-5,15,15]

param_int = [0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,0,1,
             0,1,1,]

PARAM_GROUPS = {
    'power': slice(0, 8),
    'gain': slice(8, 16),
    'gbw': slice(16, 24)
}

def generate_individual():
    num_triplets = 7
    individual = []
    
    for _ in range(num_triplets):
        individual.extend([random.uniform(1, 5), random.uniform(1, 5), random.randint(0, 20)])
    individual.extend([random.uniform(0, 1e-5), random.randint(5, 15), random.randint(5, 15)])
    individual[11]=random.randint(100,500)
    return individual

def crossover(parent1, parent2, eta=15):
    child1 = []
    child2 = []
    for i, (p1, p2) in enumerate(zip(parent1, parent2)):
        min_val = param_range_min[i]
        max_val = param_range_max[i]
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

def variation(individual, mutation_prob=0.1, eta=20):
    mutated = list(copy.deepcopy(individual))
    for i in range(len(mutated)):
        if random.random() < mutation_prob:
            x = mutated[i]
            min_val = param_range_min[i]
            max_val = param_range_max[i]
            delta = min(x - min_val, max_val - x)
            u = random.random()
            if u <= 0.5:
                delta_q = (2*u)**(1/(eta+1)) - 1
            else:
                delta_q = 1 - (2*(1-u))**(1/(eta+1))
            mutated[i] = x + delta_q * delta
            mutated[i] = max(min_val, min(mutated[i], max_val))
            if param_int[i] ==  1:
                mutated[i] = int(round(mutated[i]))
    return creator.Individual(mutated)

def get_parato_fronts(individuals):
    fronts = tools.sortLogNondominated(individuals, len(individuals))
    return fronts

def plot_3d_reference_points():
    # Éú³É3Ä¿±ê£¬Ã¿Ä¿±ê4·ÖÇøµÄ²Î¿¼µã
    num_objs = 3
    divisions = 4
    ref_points = generate_reference_points(num_objs, divisions)
    
    # ÌáÈ¡×ø±ê
    x = [point[0] for point in ref_points]
    y = [point[1] for point in ref_points]
    z = [point[2] for point in ref_points]
    
    # ´´½¨3DÍ¼
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # »æÖÆ²Î¿¼µã
    ax.scatter(x, y, z, c='r', marker='o', s=50, depthshade=True)
    
    # ÉèÖÃÖá±êÇ©ºÍ±êÌâ
    ax.set_xlabel('Objective 1', fontsize=12)
    ax.set_ylabel('Objective 2', fontsize=12)
    ax.set_zlabel('Objective 3', fontsize=12)
    ax.set_title(f'NSGA-III Reference Points (M={num_objs}, Divisions={divisions})', 
                fontsize=14, pad=20)
    
    # ÉèÖÃÖá·¶Î§²¢Ìí¼ÓÍø¸ñ
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.grid(True)
    
    # µ÷ÕûÊÓ½Ç
    ax.view_init(elev=25, azim=45)
    
    plt.show()

def generate_ref_points_from_extremes(extreme_points, num_divisions):

    ref_points = []
    
    # Éú³ÉÖØÐÄ×ø±êÏµÏÂµÄ×éºÏ
    for i in range(num_divisions + 1):
        for j in range(num_divisions + 1 - i):
            k = num_divisions - i - j
            # ¼ÆËãÖØÐÄ×ø±ê
            u = i / num_divisions
            v = j / num_divisions
            w = k / num_divisions
            
            # ¼ÆËãÈýÎ¬×ø±ê
            coord = [
                u*extreme_points[0][0] + v*extreme_points[1][0] + w*extreme_points[2][0],
                u*extreme_points[0][1] + v*extreme_points[1][1] + w*extreme_points[2][1],
                u*extreme_points[0][2] + v*extreme_points[1][2] + w*extreme_points[2][2]
            ]
                
            ref_points.append(ReferencePoint(coord))
    
    return ref_points

def plot_3d_points_with_extremes(selected_fitness, ref_points):
    """»æÖÆ°üº¬¼«ÖµµãºÍ²Î¿¼µãµÄ3DÍ¼"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    extreme_points = selected_fitness[-3:]
    # »æÖÆ²Î¿¼µã
    x = [p[0] for p in ref_points]
    y = [p[1] for p in ref_points]
    z = [p[2] for p in ref_points]
    ax.scatter(x, y, z, c='b', marker='o', s=40, label='Reference Points')
    
    # »æÖÆ¼«Öµµã
    ex_x = [p[0] for p in selected_fitness]
    ex_y = [p[1] for p in selected_fitness]
    ex_z = [p[2] for p in selected_fitness]
    ax.scatter(ex_x, ex_y, ex_z, c='r', marker='.', s=100, label='Extreme Points')

    
    # »æÖÆ¼«ÖµµãÖ®¼äµÄÁ¬Ïß
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([extreme_points[i][0], extreme_points[j][0]],
                    [extreme_points[i][1], extreme_points[j][1]],
                    [extreme_points[i][2], extreme_points[j][2]], 
                    'k--', alpha=0.3)
    
    # ÉèÖÃÍ¼ÐÎ²ÎÊý
    ax.set_xlabel('Objective 1', fontsize=12)
    ax.set_ylabel('Objective 2', fontsize=12)
    ax.set_zlabel('Objective 3', fontsize=12)
    ax.set_title('Reference Points Generated from Extreme Points', fontsize=14)
    ax.legend()
    
    # ÉèÖÃ×ø±êÖá·¶Î§
    max_range = max([max(x), max(y), max(z)])
    ax.set_xlim([0, max_range*1.1])
    ax.set_ylim([0, max_range*1.1])
    ax.set_zlim([0, max_range*1.1])
    
    # µ÷ÕûÊÓ½Ç
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    plt.show()

def extract_fitness_coordinates(individuals):
    return [
        [ind.fitness.values[0], 
         ind.fitness.values[1],
         ind.fitness.values[2]]
        for ind in individuals
        if ind.fitness.valid 
    ]

def get_plane_axis_intercepts(points):
    p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2])
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    normal = np.cross(v1, v2)
    a, b, c = normal
    d = np.dot(normal, p1)
    
    intercepts = []
    
    if a != 0:
        x_int = d / a
        intercepts.append([x_int, 0, 0])
    else:
        intercepts.append([np.nan, 0, 0])
    
    # Óë y Öá½»µã£ºx = 0, z = 0 ¡ú ½â y
    if b != 0:
        y_int = d / b
        intercepts.append([0, y_int, 0])
    else:
        intercepts.append([0, np.nan, 0])
    
    if c != 0:
        z_int = d / c
        intercepts.append([0, 0, z_int])
    else:
        intercepts.append([0, 0, np.nan])
    
    return np.array(intercepts)

def param_to_fitness_24d(individual):
    params = np.array(individual)
    
    power_terms = params[PARAM_GROUPS['power']]
    power_base = 0.2 * np.sum(power_terms**2) + 5.0
    
    cross_term1 = 0.1 * np.mean(params[PARAM_GROUPS['gain']][:2])
    
    total_power = power_base + cross_term1

    gain_params = params[PARAM_GROUPS['gain']]
    gain_main = 20 * np.log10(
        1 + 0.5 * (gain_params[0]*gain_params[1] + 
                 gain_params[2]*gain_params[3])
                )
    
    penalty_term = 0.05 * np.sum(params[PARAM_GROUPS['gbw']][::2])
    
    total_gain = gain_main - penalty_term
    
    gbw_params = params[PARAM_GROUPS['gbw']]
    gbw_main = (np.prod(gbw_params[:4])**0.25 * 1e9)
    
    limiter = 1.0 - 0.1 * np.mean(params[PARAM_GROUPS['power']][4:6])
    
    total_gbw = gbw_main * limiter
    
    return (total_power, total_gain, total_gbw)

def enhanced_fitness_24d(individual):
    x = np.array(individual)
    
    power_base = 0.3 * np.sum(x**3) 
    chaos_term = 0.2 * sum(x[i]*x[(i+3)%24]*(1-x[(i+3)%24]) for i in [0,5,10,15])
    sin_mod = 0.5 * np.sin(2*np.pi*(x[2]*x[7] + x[12]/5))
    total_power = power_base + chaos_term + sin_mod + 5
    
    tanh_term = np.tanh(0.1*(x[3] + 2*x[8] - x[13]))
    frac_term = (x[4]*x[9] + 1e-6) / (0.1 + x[14]**2)
    exp_term = 2 * np.exp(-0.5*(x[1] - x[6])**2)
    total_gain = 20 * np.log10( (abs(tanh_term * frac_term) * exp_term + 1e-6 ) )
    
    cross_prod = np.prod([x[i]+0.5 for i in [5,10,15,20]])
    damped_osc = np.sin(3*x[0]) * np.exp(-0.7*x[11])
    stoch_term = 0.3 * sum((-1)**i * x[i%24] for i in range(6))
    total_gbw = (cross_prod**0.25 + 0.5*damped_osc + stoch_term) * 1e9
    
    return (total_power, total_gain, total_gbw)

def nsga_selection(population, amp, times=10, selection=10):
    #amp = AMPNMCFEnv()
    for i in range(times):
        for ind in population:
            if not ind.fitness.valid:
                try:
                    amp._initialize_simulation()
                    amp.do_simulation(np.array(ind))
                    observation = amp._get_obs()
                    info = amp._get_info()
                    evaluated_value = (info['Power'], info['dcgain'], info['GBW'], info['phase_margin (deg)'], info['TC'], info['vos'], info['cmrrdc'], info['PSRP'], info['PSRN'], info['sr'], info['settlingTime'])
                except Exception as e:
                    print(f"Simulation failed for individual {ind}. Error: {e}")
                    evaluated_value = (float('inf'), -float('inf'), -float('inf'), -float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'), float('inf'))
                ind.fitness.values = evaluated_value

        selected_individuals = sel_nsga_iii(population, k=selection)
    return selected_individuals

def crossover_variation(individuals):
    offspring = []
    for i in range(0, len(individuals)-1, 2):
        parent1, parent2 = individuals[i], individuals[i+1]
        child1, child2 = crossover(parent1, parent2)
        offspring.append(variation(child1))
        offspring.append(variation(child2))
    return individuals + offspring

if __name__ == "__main__":

    amp = os.system("helo")
    times = 20
    divisions = 49

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    population = [creator.Individual(generate_individual()) for _ in range(64)]
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'p', 'w']
    
    for i in range(times):
        for ind in population:
            if not ind.fitness.valid:
                try:
                    amp._initialize_simulation()
                    amp.do_simulation(np.array(ind))
                    observation = amp._get_obs()
                    info = amp._get_info()
                    evaluated_value = (info['Power'], info['dcgain'], info['GBW'], info[''])
                    #evaluated_value = enhanced_fitness_24d(ind)
                except Exception as e:
                    print(f"Simulation failed for individual {ind}. Error: {e}")
                    evaluated_value = (float('inf'), -float('inf'), -float('inf'))
                ind.fitness.values = evaluated_value

        selected_individuals = sel_nsga_iii(population, k=32)
        #selected_fitness = extract_fitness_coordinates(selected_individuals)
        #ref = generate_ref_points_from_extremes(get_plane_axis_intercepts(selected_fitness[-3:]),4)
        #plot_3d_points_with_extremes(selected_fitness,ref)

        if i == times-1:
            break
        offspring = []
        for i in range(0, len(selected_individuals)-1, 2):
            parent1, parent2 = selected_individuals[i], selected_individuals[i+1]
            child1, child2 = crossover(parent1, parent2)
            offspring.append(variation(child1))
            offspring.append(variation(child2))

        population[:] = selected_individuals + offspring
        with open("population.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            for ind in population:
                row = list(ind) + list(ind.fitness.values)
                writer.writerow(row)

    fronts = get_parato_fronts(population)
    num_objectives = 3    

    if num_objectives == 2:
        fig, ax = plt.subplots()
        #ax.scatter(fitness_values[:, 0], fitness_values[:, 1], c='blue', label='Pareto Front')
        ax.set_xlabel('Objective 1 (Power)')
        ax.set_ylabel('Objective 2 (DC Gain)')
        ax.legend()
        plt.title('2D Pareto Front')
        plt.show()
    elif num_objectives >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i,front in enumerate(fronts[:6]):
            #ax.scatter(fitness_values[:, 0], fitness_values[:, 1], fitness_values[:, 2], c='black', label='Pareto Front')
            xs = [ind.fitness.values[0] for ind in front]
            ys = [ind.fitness.values[1] for ind in front]
            zs = [ind.fitness.values[2] for ind in front]
            ax.scatter(xs, ys, zs, c=colors[i%6], label=f'Pareto Front {i+1}')
        ax.set_xlabel('Power')
        ax.set_ylabel('DC Gain')
        ax.set_zlabel('GBW')
        ax.legend()
        plt.title('3D Pareto Front')
        plt.show()
    else:
        print("more num_objectives")