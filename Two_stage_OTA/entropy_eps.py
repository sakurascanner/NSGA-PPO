import numpy as np
import pandas as pd
import csv
import ast
import os
import matplotlib
print(matplotlib.get_cachedir())
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
    with open('entropy_1w.csv', 'r') as f:
        data = [float(line.strip()) for line in f]
except Exception as e:
    print(f"Error reading file: {e}")

try:
    # ¶ÁÈ¡CSVÎÄ¼þ£¬¼ÙÉèÃ¿ÐÐÊÇÒ»¸öÊýÖµ
    with open('entropy.csv', 'r') as f:
        data2 = [float(line.strip()) for line in f]
except Exception as e:
    print(f"Error reading file: {e}")

group_size = 100
group_size2 = 7
groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]
groups2 = [data2[i:i+group_size] for i in range(0, len(data2), group_size)]

# ¼ÆËãÃ¿×éµÄÆ½¾ùÖµ
averages = [sum(g)/len(g) -11 for i, g in enumerate(groups)]
averages2 = [sum(g)/len(g) for g in groups2]
#plt.plot(averaged_rewards_by_env[11][0:330],color="#FF44C1F8",ls='-',label='GCN; S/P:15/50')
#plt.plot(reward_power_650,color='green',ls='-', label='ANN; S/P:15/50')
#plt.plot(reward_nsga_suc,color='#4169E1',ls='-', label='GCN; S/P:50/15')
plt.plot(averages,color="#4BAAF8",ls='-')
#plt.plot(averages2,color="#ffacf8",ls='-')

plt.xticks(np.arange(0,110,30))
plt.yticks(np.arange(6,16,3))
plt.ylim(5,16)
plt.xlabel('Episode',fontsize=15)
plt.ylabel('Entropy',fontsize=15)
plt.tick_params(axis='both',direction='in',labelsize=12, pad=8,length=6,width=3)
plt.legend(bbox_to_anchor=(1.0,0.95))
plt.tight_layout()
plt.savefig('Entropy.eps',format='eps',dpi=300)
plt.show()