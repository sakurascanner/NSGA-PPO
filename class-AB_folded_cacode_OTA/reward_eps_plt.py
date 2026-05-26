import numpy as np
import pandas as pd
import csv
import ast
import os
import matplotlib
print(matplotlib.get_cachedir())
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter, EngFormatter

def group_average(data, group_size):
    groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]
    averages = [sum(g)/len(g) for g in groups]
    std = [np.std(g) for g in groups]
    return averages, std

reward_by_env = []

try:
    # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
    #with open('reward_nsga_success.csv', 'r') as f:
    with open('reward_64step_15pop.csv', 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            reward_list = ast.literal_eval(line[0])

            if not reward_by_env:
                reward_by_env = [[] for _ in range(len(reward_list))]

            for i, r in enumerate(reward_list):
                reward_by_env[i].append(r)
except Exception as e:
    print(f"Error reading file: {e}")

group_size = 100
averaged_env =[]
averaged_rewards_by_env = []
averaged_std_by_env = []

for env_rewards in reward_by_env[0:12]:
    avg, std = group_average(env_rewards, group_size)
    averaged_rewards_by_env.append(avg)
    averaged_std_by_env.append(std)
mean_curve = np.mean(averaged_rewards_by_env, axis=0)
std_curve = np.mean(averaged_std_by_env, axis=0)
steps = np.arange(len(mean_curve))

#plt.figure(figsize=(12, 7))
num_envs = len(reward_by_env)
#for i in range(num_envs):
#    plt.plot(averaged_rewards_by_env[i], label=f'Env {i+1}', alpha=0.7)
plt.plot(steps,mean_curve, label="Mean Reward")
poly = plt.fill_between(steps,mean_curve- std_curve,
                    mean_curve + std_curve,
                    color='blue', alpha=0.2, label='Standerd Deviation')

poly.set_rasterized(True)
plt.xticks(np.arange(0,1100,300))
plt.yticks(np.arange(-6,3.5,3))
plt.xlabel('Episode',fontsize=15)
plt.ylabel('Reward',fontsize=15)
plt.tick_params(axis='both',direction='in',labelsize=12, pad=8,length=6,width=3)
plt.legend(bbox_to_anchor=(1.0,0.25))
plt.tight_layout()
plt.savefig('MainReward.png',format='png',dpi=290)
plt.show()