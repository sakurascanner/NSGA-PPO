import torch
import numpy as np
import os
import math
import json
from tabulate import tabulate
import gymnasium as gym
import multiprocessing as mp
import uuid
import shutil
from NSGA import NSGA_Agent
from selection import sel_nsga_iii
from gymnasium import spaces

from ckt_graph import GraphAMPNMCF
from dev_params import DeviceParams
from utils import ActionNormalizer, OutputParser2
from datetime import datetime


date = datetime.today().strftime('%Y-%m-%d')
PWD = os.getcwd()
SPICE_NETLIST_DIR = f'{PWD}/simulations'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CktGraph = GraphAMPNMCF
            
class AMPNMCFEnv(gym.Env, CktGraph, DeviceParams):

    def __init__(self, reward_model=None, **kwargs):
        gym.Env.__init__(self)
        CktGraph.__init__(self)
        DeviceParams.__init__(self, self.ckt_hierarchy)

        self.CktGraph = CktGraph()
        self.NSGA_agent = NSGA_Agent()
        self.reward_model = reward_model or kwargs.get("reward_model", None)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float64)
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_shape, dtype=np.float64)
        self.delta_action = np.array([0.5, 0.5, 1,
                             0.5, 0.5, 1,
                             0.5, 0.5, 1,
                             0.5, 0.5, 10,
                             0.5, 0.5, 1,
                             0.5, 0.5, 1,
                             0.5, 0.5, 1,
                             1e-6,
                             1,
                             1])
        self.action_space_low = np.array([ 0.5, 0.5, 1, # M0(W_low,L_low,M_low)
                                        0.5, 0.5, 1,    # M8
                                        0.5, 0.5, 1,      # M10
                                        0.5, 0.5, 100,  # M11
                                        0.5, 0.5, 1,    # M17
                                        0.5, 0.5, 1,    # M21
                                        0.5, 0.5, 1,      # M23
                                        3e-6,         # Ib
                                        1,     # C0
                                        1])  # C1
        
        self.action_space_high = np.array([10, 4, 50,  # M0(W_high,L_high,M_high) 
                                        10, 4, 50,     # M8  
                                        10, 4, 50,     # M10
                                        10, 4, 500,    # M11
                                        10, 4, 50,     # M17
                                        10, 4, 50,    # M21
                                        10, 5, 50,    # M23
                                        20e-6,        # Ib  
                                        50,    # C0
                                        50])   # C1
        self.parameters_population = []
        self.initial_NSGA_population(self.NSGA_agent.population, 8)
        
        
    def initial_NSGA_population(self, population, times):
        for i in range(times):
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(simulate_individual, [(self, ind, j) for j, ind in enumerate(population)])

            for ind, info in zip(population,results):
                print(info)
                ind.fitness.values = (info['Power'][1], info['dcgain'][1], info['GBW'][1], info['phase_margin (deg)'][1], info['TC'][1], info['vos'][1], info['cmrrdc'][1], info['PSRP'][1], info['PSRN'][1], info['sr'][1], info['settlingTime'][1], info['reward'])

            selected_individuals = sel_nsga_iii(population, k=7)
            if i == times-1:
                break
            offspring = self.NSGA_agent.crossover_variation(selected_individuals)
            population[:] = selected_individuals + offspring
        
        #for ind in population:
        #    print("selected population:",np.array(ind))
        #    print("and its fitness:",ind.fitness.values,"\n")

        
    #def get_fitness(self, population, info):
    #    for ind in population:
    #        if not ind.fitness.valid:
    #            evaluated_value = (info['Power'], info['dcgain'], info['GBW'], info['phase_margin (deg)'], info['TC'], info['vos'], info['cmrrdc'], info['PSRP'], info['PSRN'], info['sr'], info['settlingTime'], info['reward'])
    #            ind.fitness.values = evaluated_value
        #selected_individuals = self.NSGA_agent.selection(selection)
        #return [np.array(ind) for ind in selected_individuals]
        
    def _initialize_simulation(self, episodes):
        self.parameters_population = [np.array(ind) for ind in self.NSGA_agent.population]
        ind = self.NSGA_agent.population
        return np.array(ind[episodes])

        #index = episodes % len(self.parameters_population)

        #self.parameters = self.parameters_population[index]

        """self.W_M0, self.L_M0, self.M_M0, \
            self.W_M8, self.L_M8, self.M_M8,\
            self.W_M10, self.L_M10, self.M_M10,\
            self.W_M11, self.L_M11, self.M_M11, \
            self.W_M17, self.L_M17, self.M_M17,\
            self.W_M21, self.L_M21, self.M_M21,\
            self.W_M23, self.L_M23, self.M_M23, \
            self.Ib,  \
            self.M_C0, \
            self.M_C1 = \
        self.parameters"""

        """Run the initial simulations."""  
        #self.do_simulation(self.parameters)
        
    #one processer
    def _do_simulation(self, action: np.array):
        unique_id = str(uuid.uuid4())[:8]
        sim_dir = f"{SPICE_NETLIST_DIR}/{unique_id}"
        os.makedirs(sim_dir, exist_ok=True)

        W_M0, L_M0, M_M0,\
        W_M8, L_M8, M_M8,\
        W_M10, L_M10, M_M10,\
        W_M11, L_M11, M_M11,\
        W_M17, L_M17, M_M17,\
        W_M21, L_M21, M_M21,\
        W_M23, L_M23, M_M23, \
        Ib, \
        M_C0, \
        M_C1 = action 
        
        M_M0 = int(M_M0)
        M_M8 = int(M_M8)
        M_M10 = int(M_M10)
        M_M11 = int(M_M11)
        M_M17 = int(M_M17)
        M_M21 = int(M_M21)
        M_M23 = int(M_M23)
        M_C0 = int(M_C0)
        M_C1 = int(M_C1)
        
        # update netlist
        try:
            # open the netlist of the testbench
            AMP_NMCF_vars = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_vars.spice', 'r')
            lines = AMP_NMCF_vars.readlines()
            lines[0] = f'.param MOSFET_0_8_W_BIASCM_PMOS={W_M0} MOSFET_0_8_L_BIASCM_PMOS={L_M0} MOSFET_0_8_M_BIASCM_PMOS={M_M0}\n'
            lines[1] = f'.param MOSFET_8_2_W_gm1_PMOS={W_M8} MOSFET_8_2_L_gm1_PMOS={L_M8} MOSFET_8_2_M_gm1_PMOS={M_M8}\n'
            lines[2] = f'.param MOSFET_10_1_W_gm2_PMOS={W_M10} MOSFET_10_1_L_gm2_PMOS={L_M10} MOSFET_10_1_M_gm2_PMOS={M_M10}\n'
            lines[3] = f'.param MOSFET_11_1_W_gmf2_PMOS={W_M11} MOSFET_11_1_L_gmf2_PMOS={L_M11} MOSFET_11_1_M_gmf2_PMOS={M_M11}\n'
            lines[4] = f'.param MOSFET_17_7_W_BIASCM_NMOS={W_M17} MOSFET_17_7_L_BIASCM_NMOS={L_M17} MOSFET_17_7_M_BIASCM_NMOS={M_M17}\n'
            lines[5] = f'.param MOSFET_21_2_W_LOAD2_NMOS={W_M21} MOSFET_21_2_L_LOAD2_NMOS={L_M21} MOSFET_21_2_M_LOAD2_NMOS={M_M21}\n'
            lines[6] = f'.param MOSFET_23_1_W_gm3_NMOS={W_M23} MOSFET_23_1_L_gm3_NMOS={L_M23} MOSFET_23_1_M_gm3_NMOS={M_M23}\n'
            lines[7] = f'.param CURRENT_0_BIAS={Ib}\n'
            lines[8] = f'.param M_C0={M_C0}\n'
            lines[9] = f'.param M_C1={M_C1}\n'
            
            Unique_var_path = os.path.join(sim_dir, "AMP_NMCF_vars.spice")
            AMP_NMCF_vars = open(Unique_var_path, 'w')
            AMP_NMCF_vars.writelines(lines)
            AMP_NMCF_vars.close()

            AMP_NMCF_ACDC = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_ACDC.cir', 'r')
            lines = AMP_NMCF_ACDC.readlines()
            lines[6] = f'.include ../../simulations/NMCF_Pin_3_HSPICE_130.txt\n'
            lines[10] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt.spice\n'
            lines[11] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical.spice\n'
            lines[12] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical__lin.spice\n'
            lines[13] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt/specialized_cells.spice\n'
            lines[21] = f'.include AMP_NMCF_vars.spice\n'
            lines[118] = f'.include ../../simulations/AMP_NMCF_dev_params.spice\n'
            Unique_ACDC_path = os.path.join(sim_dir, "AMP_NMCF_ACDC.cir")
            AMP_NMCF_ACDC = open(Unique_ACDC_path, 'w')
            AMP_NMCF_ACDC.writelines(lines)
            AMP_NMCF_ACDC.close()

            AMP_NMCF_Tran = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_Tran.cir', 'r')
            lines = AMP_NMCF_Tran.readlines()
            lines[6] = f'.include ../../simulations/NMCF_Pin_3_HSPICE_130.txt\n'
            lines[10] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt.spice\n'
            lines[11] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical.spice\n'
            lines[12] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/r+c/res_typical__cap_typical__lin.spice\n'
            lines[13] = f'.include ../../mosfet_model/sky130_pdk/libs.tech/ngspice/corners/tt/specialized_cells.spice\n'
            lines[27] = f'.include AMP_NMCF_vars.spice\n'
            lines[79] = f'.include ../../simulations/AMP_NMCF_dev_params.spice\n'
            Unique_Tran_path = os.path.join(sim_dir, "AMP_NMCF_Tran.cir")
            AMP_NMCF_Tran = open(Unique_Tran_path, 'w')
            AMP_NMCF_Tran.writelines(lines)
            AMP_NMCF_Tran.close()

            
            os.system(f'cd {sim_dir}&& ngspice -b -o AMP_NMCF_ACDC.log AMP_NMCF_ACDC.cir')
            os.system(f'cd {sim_dir}&& ngspice -b -o AMP_NMCF_Tran.log AMP_NMCF_Tran.cir')
            print('*** Simulations Done! ***')
        except:
            print('ERROR')

        return sim_dir, Ib

    #one batch
    def do_simulation(self, action):
        sim_dir, Ib = self._do_simulation(action)
        sim_results = OutputParser2(self.CktGraph, sim_dir)
        op_results = sim_results.dcop(file_name='AMP_NMCF_op')
        return op_results, sim_results, Ib

    #need batch
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        observation_batch = []
        info_batch = []
        self.parameters_population = []
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(rst_simulate_individual, [(self, ind, j) for j, ind in enumerate(self.NSGA_agent.population)])
        for observation, info in results:
            observation_batch.append(observation)
            info_batch.append(info)
        for ind in self.NSGA_agent.population:
            self.parameters_population.append(np.array(ind))
        return observation_batch, {}
    
    def close(self):
        return None
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        delta = self.delta_action * (action - 1)
        observation_batch = []
        info_batch = []
        reward_batch = []
        done_batch = []

        delta = self.delta_action * (action - 1)
        self.parameters_population = np.clip(self.parameters_population + delta,
                                        self.action_space_low,
                                        self.action_space_high)


        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(stp_simulate_individual, [(self, j, new_ind_individual) for j, new_ind_individual in enumerate(self.parameters_population)])
        for observation, info, reward, done in results:
            observation_batch.append(observation)
            info_batch.append(info)
            reward_batch.append(reward)
            done_batch.append(done)
        return observation_batch, reward_batch, done_batch, self.parameters_population, {"info_batch": info_batch},

        
        ''' run simulations '''
        #op_results, sim_results = self.do_simulation(np.array(self.parameters_population))
        
        '''get observation'''
        #observation = self._get_obs(op_results)
        #info = self._get_info(sim_results)
        #self.NSGA_agent.population.append(self.NSGA_agent.add_individual(self.parameters_population, info))

        #reward = info['reward']
          
        print(tabulate(
            [
                ['TC', info['TC'][0], self.TC_target],
                ['Power', info['Power'][0], self.Power_target],
                ['vos', info['vos'][0], self.vos_target],
                ['cmrrdc', info['cmrrdc'][0], self.cmrrdc_target],
                ['dcgain', info['dcgain'][0], self.dcgain_target],

                ['GBW', info['GBW'][0], self.GBW_target],
                ['phase_margin (deg)', info['phase_margin'][0], self.phase_margin_target],
                ['PSRP', info['PSRP'][0], self.PSRP_target],
                ['PSRN', info['PSRN'][0], self.PSRN_target],

                ['sr', info['sr'][0], self.sr_target],
                ['settlingTime', info['settlingTime'][0], self.settlingTime_target], 

                ['TC score', info['TC'][1], ''],
                ['Power score', info['Power'][1], ''],
                ['vos score', info['vos'][1], ''],
                ['cmrrdc score', info['cmrrdc'][1], ''],
                ['dcgain score', info['dcgain'][1], ''],

                ['GBW score', info['GBW'][1], ''],
                ['phase_margin (deg) score', info['phase_margin (deg)'][1], ''],
                ['PSRP score', info['PSRP'][1], ''],
                ['PSRN score', info['PSRN'][1], ''],

                ['sr score', info['sr'][1], ''],
                ['settlingTime score', info['settlingTime'][1],''],
 
                ],
            headers=['param', 'num', 'target'], tablefmt='orgtbl', numalign='right', floatfmt=".8f"
            ))

        return observation, reward, done, self.parameters_population, info 
        """self.parameters.copy()"""
        
    def _get_obs(self, op_results, Ib):
        # pick some .OP params from the dict:
        try:
            f = open(f'{SPICE_NETLIST_DIR}/AMP_NMCF_op_mean_std.json')
            op_mean_std = json.load(f)
            op_mean = op_mean_std['OP_M_mean']
            op_std = op_mean_std['OP_M_std']
            op_mean = np.array([op_mean['id'], op_mean['gm'], op_mean['gds'], op_mean['vth'], op_mean['vdsat'], op_mean['vds'], op_mean['vgs']])
            op_std = np.array([op_std['id'], op_std['gm'], op_std['gds'], op_std['vth'], op_std['vdsat'], op_std['vds'], op_std['vgs']])
        except:
            print('You need to run <_random_op_sims> to generate mean and std for transistor .OP parameters')
        
        OP_M0 = op_results['M0']
        OP_M0_norm = (np.array([OP_M0['id'],
                                OP_M0['gm'],
                                OP_M0['gds'],
                                OP_M0['vth'],
                                OP_M0['vdsat'],
                                OP_M0['vds'],
                                OP_M0['vgs']
                                ]) - op_mean)/op_std
        OP_M1 = op_results['M1']
        OP_M1_norm = (np.array([OP_M1['id'],
                                OP_M1['gm'],
                                OP_M1['gds'],
                                OP_M1['vth'],
                                OP_M1['vdsat'],
                                OP_M1['vds'],
                                OP_M1['vgs']
                                ]) - op_mean)/op_std
        OP_M2 = op_results['M2']
        OP_M2_norm = (np.array([OP_M2['id'],
                                OP_M2['gm'],
                                OP_M2['gds'],
                                OP_M2['vth'],
                                OP_M2['vdsat'],
                                OP_M2['vds'],
                                OP_M2['vgs']
                                ]) - op_mean)/op_std
        OP_M3 = op_results['M3']
        OP_M3_norm = (np.abs([OP_M3['id'],
                                OP_M3['gm'],
                                OP_M3['gds'],
                                OP_M3['vth'],
                                OP_M3['vdsat'],
                                OP_M3['vds'],
                                OP_M3['vgs']
                                ]) - op_mean)/op_std
        OP_M4 = op_results['M4']
        OP_M4_norm = (np.abs([OP_M4['id'],
                                OP_M4['gm'],
                                OP_M4['gds'],
                                OP_M4['vth'],
                                OP_M4['vdsat'],
                                OP_M4['vds'],
                                OP_M4['vgs']
                                ]) - op_mean)/op_std
        OP_M5 = op_results['M5']
        OP_M5_norm = (np.abs([OP_M5['id'],
                                OP_M5['gm'],
                                OP_M5['gds'],
                                OP_M5['vth'],
                                OP_M5['vdsat'],
                                OP_M5['vds'],
                                OP_M5['vgs']
                                ]) - op_mean)/op_std
        OP_M6 = op_results['M6']
        OP_M6_norm = (np.array([OP_M6['id'],
                                OP_M6['gm'],
                                OP_M6['gds'],
                                OP_M6['vth'],
                                OP_M6['vdsat'],
                                OP_M6['vds'],
                                OP_M6['vgs']
                                ]) - op_mean)/op_std
        OP_M7 = op_results['M7']
        OP_M7_norm = (np.array([OP_M7['id'],
                                OP_M7['gm'],
                                OP_M7['gds'],
                                OP_M7['vth'],
                                OP_M7['vdsat'],
                                OP_M7['vds'],
                                OP_M7['vgs']
                                ]) - op_mean)/op_std
        OP_M8 = op_results['M8']
        OP_M8_norm = (np.array([OP_M8['id'],
                                OP_M8['gm'],
                                OP_M8['gds'],
                                OP_M8['vth'],
                                OP_M8['vdsat'],
                                OP_M8['vds'],
                                OP_M8['vgs']
                                ]) - op_mean)/op_std
        OP_M9 = op_results['M9']
        OP_M9_norm = (np.array([OP_M9['id'],
                                OP_M9['gm'],
                                OP_M9['gds'],
                                OP_M9['vth'],
                                OP_M9['vdsat'],
                                OP_M9['vds'],
                                OP_M9['vgs']
                                ]) - op_mean)/op_std
        OP_M10 = op_results['M10']
        OP_M10_norm = (np.array([OP_M10['id'],
                                OP_M10['gm'],
                                OP_M10['gds'],
                                OP_M10['vth'],
                                OP_M10['vdsat'],
                                OP_M10['vds'],
                                OP_M10['vgs']
                                ]) - op_mean)/op_std
        OP_M11 = op_results['M11']
        OP_M11_norm = (np.array([OP_M11['id'],
                                OP_M11['gm'],
                                OP_M11['gds'],
                                OP_M11['vth'],
                                OP_M11['vdsat'],
                                OP_M11['vds'],
                                OP_M11['vgs']
                                ]) - op_mean)/op_std
        OP_M12 = op_results['M12']
        OP_M12_norm = (np.array([OP_M12['id'],
                                OP_M12['gm'],
                                OP_M12['gds'],
                                OP_M12['vth'],
                                OP_M12['vdsat'],
                                OP_M12['vds'],
                                OP_M12['vgs']
                                ]) - op_mean)/op_std
        OP_M13 = op_results['M13']
        OP_M13_norm = (np.array([OP_M13['id'],
                                OP_M13['gm'],
                                OP_M13['gds'],
                                OP_M13['vth'],
                                OP_M13['vdsat'],
                                OP_M13['vds'],
                                OP_M13['vgs']
                                ]) - op_mean)/op_std
        OP_M14 = op_results['M14']
        OP_M14_norm = (np.array([OP_M14['id'],
                                OP_M14['gm'],
                                OP_M14['gds'],
                                OP_M14['vth'],
                                OP_M14['vdsat'],
                                OP_M14['vds'],
                                OP_M14['vgs']
                                ]) - op_mean)/op_std
        OP_M15 = op_results['M15']
        OP_M15_norm = (np.array([OP_M15['id'],
                                OP_M15['gm'],
                                OP_M15['gds'],
                                OP_M15['vth'],
                                OP_M15['vdsat'],
                                OP_M15['vds'],
                                OP_M15['vgs']
                                ]) - op_mean)/op_std
        OP_M16 = op_results['M16']
        OP_M16_norm = (np.array([OP_M16['id'],
                                OP_M16['gm'],
                                OP_M16['gds'],
                                OP_M16['vth'],
                                OP_M16['vdsat'],
                                OP_M16['vds'],
                                OP_M16['vgs']
                                ]) - op_mean)/op_std
        OP_M17 = op_results['M17']
        OP_M17_norm = (np.array([OP_M17['id'],
                                OP_M17['gm'],
                                OP_M17['gds'],
                                OP_M17['vth'],
                                OP_M17['vdsat'],
                                OP_M17['vds'],
                                OP_M17['vgs']
                                ]) - op_mean)/op_std
        OP_M18 = op_results['M18']
        OP_M18_norm = (np.array([OP_M18['id'],
                                OP_M18['gm'],
                                OP_M18['gds'],
                                OP_M18['vth'],
                                OP_M18['vdsat'],
                                OP_M18['vds'],
                                OP_M18['vgs']
                                ]) - op_mean)/op_std
        OP_M19 = op_results['M19']
        OP_M19_norm = (np.array([OP_M19['id'],
                                OP_M19['gm'],
                                OP_M19['gds'],
                                OP_M19['vth'],
                                OP_M19['vdsat'],
                                OP_M19['vds'],
                                OP_M19['vgs']
                                ]) - op_mean)/op_std
        OP_M20 = op_results['M20']
        OP_M20_norm = (np.array([OP_M20['id'],
                                OP_M20['gm'],
                                OP_M20['gds'],
                                OP_M20['vth'],
                                OP_M20['vdsat'],
                                OP_M20['vds'],
                                OP_M20['vgs']
                                ]) - op_mean)/op_std
        OP_M21 = op_results['M21']
        OP_M21_norm = (np.array([OP_M21['id'],
                                OP_M21['gm'],
                                OP_M21['gds'],
                                OP_M21['vth'],
                                OP_M21['vdsat'],
                                OP_M21['vds'],
                                OP_M21['vgs']
                                ]) - op_mean)/op_std
        OP_M22 = op_results['M22']
        OP_M22_norm = (np.array([OP_M22['id'],
                                OP_M22['gm'],
                                OP_M22['gds'],
                                OP_M22['vth'],
                                OP_M22['vdsat'],
                                OP_M22['vds'],
                                OP_M22['vgs']
                                ]) - op_mean)/op_std
        OP_M23 = op_results['M23']
        OP_M23_norm = (np.array([OP_M23['id'],
                                OP_M23['gm'],
                                OP_M23['gds'],
                                OP_M23['vth'],
                                OP_M23['vdsat'],
                                OP_M23['vds'],
                                OP_M23['vgs']
                                ]) - op_mean)/op_std
        # it is not straightforward to extract resistance info from sky130 resistor, using the following approximation instead
        # normalize all passive components
        OP_C0_norm = ActionNormalizer(action_space_low=self.C0_low, action_space_high=self.C0_high).reverse_action(op_results['C0']['c']) # convert to (-1, 1)
        OP_C1_norm = ActionNormalizer(action_space_low=self.C1_low, action_space_high=self.C1_high).reverse_action(op_results['C1']['c']) # convert to (-1, 1)
        
        # state shall be in the order of node (node0, node1, ...)
        observation = np.array([
                               [0,0,0,0,      0,      OP_M0_norm[0],OP_M0_norm[1],OP_M0_norm[2],OP_M0_norm[3],OP_M0_norm[4],OP_M0_norm[5],OP_M0_norm[6]],
                               [0,0,0,0,      0,     OP_M1_norm[0],OP_M1_norm[1],OP_M1_norm[2],OP_M1_norm[3],OP_M1_norm[4],OP_M1_norm[5],OP_M1_norm[6]],
                               [0,0,0,0,      0,      OP_M2_norm[0],OP_M2_norm[1],OP_M2_norm[2],OP_M2_norm[3],OP_M2_norm[4],OP_M2_norm[5],OP_M2_norm[6]],
                               [0,0,0,0,      0,      OP_M3_norm[0],OP_M3_norm[1],OP_M3_norm[2],OP_M3_norm[3],OP_M3_norm[4],OP_M3_norm[5],OP_M3_norm[6]],
                               [0,0,0,0,      0,      OP_M4_norm[0],OP_M4_norm[1],OP_M4_norm[2],OP_M4_norm[3],OP_M4_norm[4],OP_M4_norm[5],OP_M4_norm[6]],
                               [0,0,0,0,      0,      OP_M5_norm[0],OP_M5_norm[1],OP_M5_norm[2],OP_M5_norm[3],OP_M5_norm[4],OP_M5_norm[5],OP_M5_norm[6]],
                               [0,0,0,0,      0,      OP_M6_norm[0],OP_M6_norm[1],OP_M6_norm[2],OP_M6_norm[3],OP_M6_norm[4],OP_M6_norm[5],OP_M6_norm[6]],
                               [0,0,0,0,      0,      OP_M7_norm[0],OP_M7_norm[1],OP_M7_norm[2],OP_M7_norm[3],OP_M7_norm[4],OP_M7_norm[5],OP_M7_norm[6]],
                               [0,0,0,0,      0,      OP_M8_norm[0],OP_M8_norm[1],OP_M8_norm[2],OP_M8_norm[3],OP_M8_norm[4],OP_M8_norm[5],OP_M8_norm[6]],
                               [0,0,0,0,      0,      OP_M9_norm[0],OP_M9_norm[1],OP_M9_norm[2],OP_M9_norm[3],OP_M9_norm[4],OP_M9_norm[5],OP_M9_norm[6]],
                               [0,0,0,0,      0,      OP_M10_norm[0],OP_M10_norm[1],OP_M10_norm[2],OP_M10_norm[3],OP_M10_norm[4],OP_M10_norm[5],OP_M10_norm[6]],
                               [0,0,0,0,      0,      OP_M11_norm[0],OP_M11_norm[1],OP_M11_norm[2],OP_M11_norm[3],OP_M11_norm[4],OP_M11_norm[5],OP_M11_norm[6]],
                               [0,0,0,0,      0,      OP_M12_norm[0],OP_M12_norm[1],OP_M12_norm[2],OP_M12_norm[3],OP_M12_norm[4],OP_M12_norm[5],OP_M12_norm[6]],
                               [0,0,0,0,      0,      OP_M13_norm[0],OP_M13_norm[1],OP_M13_norm[2],OP_M13_norm[3],OP_M13_norm[4],OP_M13_norm[5],OP_M13_norm[6]],
                               [0,0,0,0,      0,      OP_M14_norm[0],OP_M14_norm[1],OP_M14_norm[2],OP_M14_norm[3],OP_M14_norm[4],OP_M14_norm[5],OP_M14_norm[6]],
                               [0,0,0,0,      0,      OP_M15_norm[0],OP_M15_norm[1],OP_M15_norm[2],OP_M15_norm[3],OP_M15_norm[4],OP_M15_norm[5],OP_M15_norm[6]],
                               [0,0,0,0,      0,      OP_M16_norm[0],OP_M16_norm[1],OP_M16_norm[2],OP_M16_norm[3],OP_M16_norm[4],OP_M16_norm[5],OP_M16_norm[6]],
                               [0,0,0,0,      0,      OP_M17_norm[0],OP_M17_norm[1],OP_M17_norm[2],OP_M17_norm[3],OP_M17_norm[4],OP_M17_norm[5],OP_M17_norm[6]],     
                               [0,0,0,0,      0,      OP_M18_norm[0],OP_M18_norm[1],OP_M18_norm[2],OP_M18_norm[3],OP_M18_norm[4],OP_M18_norm[5],OP_M18_norm[6]],
                               [0,0,0,0,      0,      OP_M19_norm[0],OP_M19_norm[1],OP_M19_norm[2],OP_M19_norm[3],OP_M19_norm[4],OP_M19_norm[5],OP_M19_norm[6]],
                               [0,0,0,0,      0,      OP_M20_norm[0],OP_M20_norm[1],OP_M20_norm[2],OP_M20_norm[3],OP_M20_norm[4],OP_M20_norm[5],OP_M20_norm[6]],
                               [0,0,0,0,      0,      OP_M21_norm[0],OP_M21_norm[1],OP_M21_norm[2],OP_M21_norm[3],OP_M21_norm[4],OP_M21_norm[5],OP_M21_norm[6]],
                               [0,0,0,0,      0,      OP_M22_norm[0],OP_M22_norm[1],OP_M22_norm[2],OP_M22_norm[3],OP_M22_norm[4],OP_M22_norm[5],OP_M22_norm[6]],
                               [0,0,0,0,      0,      OP_M23_norm[0],OP_M23_norm[1],OP_M23_norm[2],OP_M23_norm[3],OP_M23_norm[4],OP_M23_norm[5],OP_M23_norm[6]],
                               
                               [self.Vdd,0,0,0,0,      0,0,0,0,0,0,0],
                               [0,self.GND,0,0,0,      0,0,0,0,0,0,0],
                               [0,0,Ib,0,0,       0,0,0,0,0,0,0],
                               [0,0,0,OP_C0_norm,0,    0,0,0,0,0,0,0],        
                               [0,0,0,0,OP_C1_norm,       0,0,0,0,0,0,0],                              
                               ])
        # clip the obs for better regularization
        observation = np.clip(observation, -5, 5)
        
        return observation 

    def _get_info(self, sim_results):
        '''Evaluate the performance'''
        ''' DC '''
        dc_results = sim_results.dc(file_name='AMP_NMCF_ACDC_DC')
        TC = dc_results[1][1]
        Power = dc_results[2][1]
        vos_1 = dc_results[3][1]
        vos = abs(vos_1)
             
        TC_score = np.max(np.min([(self.TC_target - TC) / (self.TC_target + TC), 0]),-1)
        Power_score = np.max(np.min([(self.Power_target - Power) / (self.Power_target + Power), 0]),-1)
        vos_score = np.max(np.min([(self.vos_target - vos) / (self.vos_target + vos), 0]),-1)

        ''' AC '''
        ac_results = sim_results.ac(file_name='AMP_NMCF_ACDC_AC')
        cmrrdc = ac_results[1][1]
        if cmrrdc > 0 :
            cmrrdc_score = -1
        else : 
            cmrrdc_score = np.max(np.min([(cmrrdc - self.cmrrdc_target) / (cmrrdc + self.cmrrdc_target), 0]),-1)
            if cmrrdc < self.cmrrdc_target:
                cmrrdc_score = 0

        PSRP = ac_results[2][1]
        if PSRP > 0 :
            PSRP_score = -1
        else : 
            PSRP_score = np.max(np.min([(PSRP - self.PSRP_target) / (PSRP + self.PSRP_target), 0]),-1)
            if PSRP < self.PSRP_target:
                PSRP_score = 0

        PSRN = ac_results[3][1]
        if PSRN > 0 :
            PSRN_score = -1
        else : 
            PSRN_score = np.max(np.min([(PSRN - self.PSRN_target) / (PSRN + self.PSRN_target), 0]),-1)
            if PSRN < self.PSRN_target:
                PSRN_score = 0

        dcgain = ac_results[4][1]
        if dcgain > 0 :
            try:
                dcgain_score = np.clip((dcgain - self.dcgain_target) / (dcgain + self.dcgain_target),-1,0)
                GBW_PM_results = sim_results.GBW_PM(file_name='AMP_NMCF_ACDC_GBW_PM')
                GBW = GBW_PM_results[1][1]
                GBW_score = np.clip((GBW - self.GBW_target) / (GBW + self.GBW_target),-1,0)
                phase_margin = GBW_PM_results[2][1]
                phase_margin_score = np.clip((phase_margin - self.phase_margin_target) / (phase_margin + self.phase_margin_target),-1,0)
            except: 
                if phase_margin > 180 or phase_margin < 0:
                    phase_margin = 0
        else :
            dcgain_score = -1
            GBW = 0
            GBW_score = -1
            phase_margin = 0
            phase_margin_score = -1
      
        """ Tran """
        tran_results = sim_results.tran(file_name='AMP_NMCF_Tran')
        sr_rise = tran_results[1][1]
        sr_fall = tran_results[2][1]
        sr = (sr_rise + sr_fall) / 2 
        sr_score = np.max(np.min([(sr - self.sr_target) / (sr + self.sr_target), 0]),-1)

        """ setting_time """
        meas = {}
        d0 = 0.01
        # path = './benchmarks/TB_Amplifier_ACDC/'
        time_data, vin_data, vout_data = sim_results.extract_tran_data(file_name='tran.dat')
        if time_data is None:
            return None
        d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n = sim_results.analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
        d0_settle = abs(d0_settle)
        d1_settle = abs(d1_settle)
        d2_settle = abs(d2_settle)
        SR_n = abs(SR_n)
        SR_p = abs(SR_p)
        settlingTime_p = abs(settling_time_p)
        settlingTime_n = abs(settling_time_n)
    
        if math.isnan(d0_settle):
            d0_settle = 10
    
        if math.isnan(d1_settle) or math.isnan(d2_settle) :
            if math.isnan(d1_settle):
                d0_settle += 10
            if math.isnan(d2_settle):
                d0_settle += 10
            d_settle = d0_settle
        else:
            d_settle = max(d0_settle, d1_settle, d2_settle)
    
        if math.isnan(SR_p) or math.isnan(SR_n) :
            SR = -d_settle
        else:
            SR = min(SR_p,SR_n)
    
        if math.isnan(settlingTime_p) or math.isnan(settlingTime_n) :
            settlingTime = d_settle
        else:
            settlingTime = max(settlingTime_p, settlingTime_n)
        
        meas['d_settle'] = d_settle
        meas['SR'] = SR
        meas['settlingTime'] = settlingTime
        settlingTime_score = np.max(np.min([(self.settlingTime_target - settlingTime) / (self.settlingTime_target + settlingTime), 0]),-1)

        if sr_score < -1:
            sr_score = -1

        shutil.rmtree(sim_results.path)

    
        """ Total reward """
        reward = TC_score + Power_score + vos_score + cmrrdc_score + \
                      dcgain_score + GBW_score + phase_margin_score + PSRP_score + \
                      PSRN_score + sr_score + settlingTime_score 
                                        
        return {
                'TC': [TC, TC_score], 
                'Power': [Power, Power_score], 
                'vos': [vos, vos_score], 
                'cmrrdc': [cmrrdc, cmrrdc_score], 
                'dcgain': [dcgain, dcgain_score], 

                'GBW': [GBW, GBW_score], 
                'phase_margin (deg)': [phase_margin, phase_margin_score], 
                'PSRP': [PSRP, PSRP_score], 
                'PSRN': [PSRN, PSRN_score], 

                'sr': [sr, sr_score], 
                'settlingTime': [settlingTime, settlingTime_score],
                'reward': reward
            }


    def _init_random_sim(self, max_sims=100):
        '''
        
        This is NOT the same as the random step in the agent, here is basically 
        doing some completely random design variables selection for generating
        some device parameters for calculating the mean and variance for each
        .OP device parameters (getting a statistical idea of, how each ckt parameter's range is like'), 
        so that you can do the normalization for the state representations later.
    
        '''
        random_op_count = 0
        OP_M_lists = []
        OP_R_lists = []
        OP_C_lists = []
        OP_V_lists = []
        OP_I_lists = []
        
        while random_op_count <= max_sims :
            print(f'* simulation #{random_op_count} *')
            action = np.random.uniform(self.action_space_low, self.action_space_high, self.action_dim) 
            print(f'action: {action}')
            sim_dir = self._do_simulation(action)
    
            sim_results = OutputParser2(self.CktGraph, sim_dir)
            op_results = sim_results.dcop(file_name='AMP_NMCF_op')
            
            OP_M_list = []
            OP_R_list = []
            OP_C_list = []
            OP_V_list = []
            OP_I_list = []

            for key in list(op_results):
                if key[0] == 'M' or key[0] == 'm':
                    OP_M = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_M_list.append(OP_M)
                elif key[0] == 'R' or key[0] == 'r':
                    OP_R = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_R_list.append(OP_R)
                elif key[0] == 'C' or key[0] == 'c':
                    OP_C = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_C_list.append(OP_C)   
                elif key[0] == 'V' or key[0] == 'v':
                    OP_V = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_V_list.append(OP_V)
                elif key[0] == 'I' or key[0] == 'i':
                    OP_I = np.array([op_results[key][f'{item}'] for item in list(op_results[key])])    
                    OP_I_list.append(OP_I)   
                else:
                    None
                    
            OP_M_list = np.array(OP_M_list)
            OP_R_list = np.array(OP_R_list)
            OP_C_list = np.array(OP_C_list)
            OP_V_list = np.array(OP_V_list)
            OP_I_list = np.array(OP_I_list)
                        
            OP_M_lists.append(OP_M_list)
            OP_R_lists.append(OP_R_list)
            OP_C_lists.append(OP_C_list)
            OP_V_lists.append(OP_V_list)
            OP_I_lists.append(OP_I_list)
            
            random_op_count = random_op_count + 1

        OP_M_lists = np.array(OP_M_lists)
        OP_R_lists = np.array(OP_R_lists)
        OP_C_lists = np.array(OP_C_lists)
        OP_V_lists = np.array(OP_V_lists)
        OP_I_lists = np.array(OP_I_lists)
        
        if OP_M_lists.size != 0:
            OP_M_mean = np.mean(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]), axis=0)
            OP_M_std = np.std(OP_M_lists.reshape(-1, OP_M_lists.shape[-1]),axis=0)
            OP_M_mean_dict = {}
            OP_M_std_dict = {}
            for idx, key in enumerate(self.params_mos):
                OP_M_mean_dict[key] = OP_M_mean[idx]
                OP_M_std_dict[key] = OP_M_std[idx]
        
        if OP_R_lists.size != 0:
            OP_R_mean = np.mean(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]), axis=0)
            OP_R_std = np.std(OP_R_lists.reshape(-1, OP_R_lists.shape[-1]),axis=0)
            OP_R_mean_dict = {}
            OP_R_std_dict = {}
            for idx, key in enumerate(self. params_r):
                OP_R_mean_dict[key] = OP_R_mean[idx]
                OP_R_std_dict[key] = OP_R_std[idx]
                
        if OP_C_lists.size != 0:
            OP_C_mean = np.mean(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]), axis=0)
            OP_C_std = np.std(OP_C_lists.reshape(-1, OP_C_lists.shape[-1]),axis=0)
            OP_C_mean_dict = {}
            OP_C_std_dict = {}
            for idx, key in enumerate(self.params_c):
                OP_C_mean_dict[key] = OP_C_mean[idx]
                OP_C_std_dict[key] = OP_C_std[idx]     
                
        if OP_V_lists.size != 0:
            OP_V_mean = np.mean(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]), axis=0)
            OP_V_std = np.std(OP_V_lists.reshape(-1, OP_V_lists.shape[-1]),axis=0)
            OP_V_mean_dict = {}
            OP_V_std_dict = {}
            for idx, key in enumerate(self.params_v):
                OP_V_mean_dict[key] = OP_V_mean[idx]
                OP_V_std_dict[key] = OP_V_std[idx]
        
        if OP_I_lists.size != 0:
            OP_I_mean = np.mean(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]), axis=0)
            OP_I_std = np.std(OP_I_lists.reshape(-1, OP_I_lists.shape[-1]),axis=0)
            OP_I_mean_dict = {}
            OP_I_std_dict = {}
            for idx, key in enumerate(self.params_i):
                OP_I_mean_dict[key] = OP_I_mean[idx]
                OP_I_std_dict[key] = OP_I_std[idx]

        self.OP_M_mean_std = {
            'OP_M_mean': OP_M_mean_dict,         
            'OP_M_std': OP_M_std_dict
            }

        with open(f'{sim_dir}/AMP_NMCF_op_mean_std.json','w') as file:
            json.dump(self.OP_M_mean_std, file)


def simulate_individual(args):
    env, ind, idx = args
    try:
        #env._initialize_simulation(idx)
        _, sim_results, _ = env.do_simulation(np.array(ind))
        info = env._get_info(sim_results)
        return info
    except:
        return  {
                'TC': [0,-0.999],
                'Power': [0,-0.999],
                'vos': [0,-0.999],
                'cmrrdc': [0,-0.999],
                'dcgain': [0,-0.999],

                'GBW': [0,-0.999],
                'phase_margin (deg)': [0,-0.999],
                'PSRP': [0,-0.999],
                'PSRN': [0,-0.999],

                'sr': [0,-0.999],
                'settlingTime': [0,-0.999],
                'reward': -11.0
            }

def rst_simulate_individual(args):
    env, ind, idx = args
    #env._initialize_simulation(idx)
    try:
        op_results, sim_results, Ib = env.do_simulation(np.array(ind))
        observation = env._get_obs(op_results, Ib)
        info = env._get_info(sim_results)
        return observation, info
    except:
        return  observation, {
                'TC': [0,-0.999],
                'Power': [0,-0.999],
                'vos': [0,-0.999],
                'cmrrdc': [0,-0.999],
                'dcgain': [0,-0.999],

                'GBW': [0,-0.999],
                'phase_margin (deg)': [0,-0.999],
                'PSRP': [0,-0.999],
                'PSRN': [0,-0.999],

                'sr': [0,-0.999],
                'settlingTime': [0,-0.999],
                'reward': -11.0
            }


def stp_simulate_individual(args):
    env, idx, new_ind_individual = args
    ''' run simulations '''
    #env._initialize_simulation(idx)
    op_results, sim_results, Ib = env.do_simulation(np.array(new_ind_individual))
        
    '''get observation'''
    observation = env._get_obs(op_results, Ib)
    try:
        info = env._get_info(sim_results)
    except:
        info = {
                'TC': [0,-0.999],
                'Power': [0,-0.999],
                'vos': [0,-0.999],
                'cmrrdc': [0,-0.999],
                'dcgain': [0,-0.999],

                'GBW': [0,-0.999],
                'phase_margin (deg)': [0,-0.999],
                'PSRP': [0,-0.999],
                'PSRN': [0,-0.999],

                'sr': [0,-0.999],
                'settlingTime': [0,-0.999],
                'reward': -11.0
            }
    env.NSGA_agent.population.append(env.NSGA_agent.add_individual(new_ind_individual, info))

    reward = info['reward']

    if reward >= 0:
        reward = 10
        done = True
    else:
        done = False

    return observation, info, reward, done