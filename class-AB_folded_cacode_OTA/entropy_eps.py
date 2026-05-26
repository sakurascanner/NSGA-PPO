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
    with open('entropy.csv', 'r') as f:
        data = [float(line.strip()) for line in f]
except Exception as e:
    print(f"Error reading file: {e}")

group_size = 100
groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]

# ¼ÆËãÃ¿×éµÄÆ½¾ùÖµ
averages = [sum(g)/len(g) for g in groups]


plt.plot(averages,color="#4BAAF8",ls='-', label='Average Entropy')

#plt.xticks(np.arange(0,1100,300))
#plt.yticks(np.arange(6,27,5))
#plt.ylim(5,27)
plt.xlabel('Episode',fontsize=15)
plt.ylabel('Entropy',fontsize=15)
plt.tick_params(axis='both',direction='in',labelsize=12, pad=8,length=6,width=3)
plt.legend(bbox_to_anchor=(1.0,0.95))
plt.tight_layout()
#plt.savefig('Entropy.eps',format='eps',dpi=300)
plt.show()