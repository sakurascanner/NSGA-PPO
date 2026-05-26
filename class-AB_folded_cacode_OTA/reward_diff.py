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

def get_reward_from_csv(file, group_size):
    try:
        with open(file,'r') as f:
            data = [float(line.strip()) for line in f]
        for i in range(len(data)):
            if data[i] < -10:
                data[i] = data[i] / 10
        groups = [data[i:i+group_size] for i in range(0, len(data), 64)]
        averages = [sum(g)/len(g) for g in groups]
    except Exception as e:
        print(f"Error reading file: {e}")
    return averages


reward_by_env = []

try:
    #with open('reward_64step_15pop.csv', 'r') as f:
    with open('reward_used.csv', 'r') as f:
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

num_envs = len(reward_by_env)
reward_power_650 = get_reward_from_csv("reward_power_650.csv",group_size=group_size)
#can
reward_PPO_ann = get_reward_from_csv("reward_PPO_ann.csv",group_size=group_size)
#cant
reward_entropy_not_in = get_reward_from_csv("reward_entropy_not_in.csv",group_size=group_size)
#cant
reward_nsga_suc = get_reward_from_csv("reward_nsga_success.csv",group_size=group_size)
#can
reward_continue_start = get_reward_from_csv("reward_continue_start.csv",group_size=group_size)
#cant
reward_STEPS_3_reward_worse = get_reward_from_csv("reward_STEPS_3_reward_worse.csv",group_size=group_size)
#cant
reward_STEPS_10 = get_reward_from_csv("reward_STEPS_10.csv",group_size=group_size)
#cant
reward_STEPS_50 = get_reward_from_csv("reward_STEPS_50.csv",group_size=group_size)
#cant
#reward = get_reward_from_csv("reward.csv",group_size=group_size)


plt.plot(averaged_rewards_by_env[11][0:330],color="#FF44C1F8",ls='-',label='GCN; S/P:15/50')
plt.plot(reward_power_650,color='green',ls='-', label='ANN; S/P:15/50')
plt.plot(reward_nsga_suc,color='#4169E1',ls='-', label='GCN; S/P:50/15')
plt.plot(reward_STEPS_50,color='#8B6969',ls='-', label='ANN; S/P:50/15')

plt.xticks(np.arange(0,330,100))
plt.yticks(np.arange(-6,1,3))
plt.xlabel('Episode',fontsize=15)
plt.ylabel('Reward',fontsize=15)
plt.tick_params(axis='both',direction='in',labelsize=12, pad=8,length=6,width=3)
plt.legend(bbox_to_anchor=(1.0,0.35))
plt.tight_layout()
plt.savefig('RewardCompare.eps',format='eps',dpi=300)
plt.show()