import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mtick

# ==========================================
# 1. ºËÐÄ¹¤¾ßº¯Êý (±£³Ö²»±ä)
# ==========================================
def group_average(data, group_size):
    """
    ±£ÁôÄãÌá¹©µÄÆ½»¬º¯Êý£º½«Êý¾Ý·Ö¿éÈ¡Æ½¾ùºÍ±ê×¼²î
    """
    # ½Ø¶Ï¶àÓàÊý¾Ý
    n = len(data)
    n_groups = n // group_size
    data = data[:n_groups * group_size]
    
    groups = [data[i:i+group_size] for i in range(0, len(data), group_size)]
    averages = [np.mean(g) for g in groups]
    std = [np.std(g) for g in groups]
    return np.array(averages), np.array(std)

def apply_custom_style(ax):
    """
    Ó¦ÓÃÄãÌá¹©µÄÌØ¶¨×ø±êÖáÑùÊ½
    """
    ax.tick_params(axis='both', direction='in', labelsize=15, pad=8, length=6, width=3)
    # ÉèÖÃ±ß¿ò´ÖÏ¸
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)

def load_batch_data(filename, max_steps=4000, batch_size=7):
    """
    ¶ÁÈ¡Êý¾Ý²¢´¦ÀíÎª Batch ¾ùÖµ
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return np.array([])
    
    try:
        df = pd.read_csv(filename, header=None, on_bad_lines='skip')
        raw_data = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
        
        if len(raw_data) > max_steps:
            raw_data = raw_data[:max_steps]
            
        # ¼ÆËã Batch Mean (Ã¿7¸öµãËãÒ»¸öÆ½¾ùÖµ)
        n_batches = len(raw_data) // batch_size
        raw_data = raw_data[:n_batches * batch_size]
        batch_data = raw_data.reshape(-1, batch_size).mean(axis=1)
        
        return batch_data
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return np.array([])

# ==========================================
# 2. ÅäÖÃÓëÊý¾Ý¶ÁÈ¡
# ==========================================
files_config = [
    ("reward.csv", "Pure PPO (Random Init)", "blue"), 
    ("nsga3_population_log_0.csv", "NSGA-III (Baseline)", "green"),
    ("reward_1w.csv", "Ours (GNN + Transfer)", "red"), 
]

MAX_STEPS = 4000
BATCH_SIZE = 3
GROUP_SIZE = 15 
STEPS_PER_EPISODE = 40  # [ÐÂÔö] ¶¨Òå 1 Episode = 100 Steps

# ¶ÁÈ¡ËùÓÐÊý¾Ý
data_store = {}
for fname, label, color in files_config:
    data_store[label] = load_batch_data(fname, MAX_STEPS, BATCH_SIZE)

# ==========================================
# 3. »æÍ¼ 1: Average Reward Comparison
# ==========================================
plt.figure(figsize=(10, 7.3))
ax = plt.gca()

for fname, label, color in files_config:
    raw_batches = data_store[label]
    if len(raw_batches) == 0: continue
    
    # Æ½»¬Êý¾Ý
    means, stds = group_average(raw_batches, GROUP_SIZE)
    
    # [ÐÞ¸Äµã] ¼ÆËã X Öá×ø±ê (Episodes)
    # 1. ¼ÆËãÃ¿¸öÊý¾Ýµã´ú±íµÄ×Ü Steps Êý
    steps_per_point = GROUP_SIZE * BATCH_SIZE
    total_steps = np.arange(1, len(means) + 1) * steps_per_point
    
    # 2. ½« Steps ×ª»»Îª Episodes
    episodes = total_steps / STEPS_PER_EPISODE
    
    # »æÍ¼
    plt.plot(episodes, means, label=label, color=color, linewidth=2)
    
    # »æÖÆÒõÓ°
    poly = plt.fill_between(episodes, means - stds, means + stds,
                            color=color, alpha=0.15)
    poly.set_rasterized(True)

# Ó¦ÓÃÑùÊ½
apply_custom_style(ax)

# [ÐÞ¸Äµã] ×ø±êÖá±êÇ©
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Average Reward', fontsize=18)

# [¿ÉÑ¡] ÊÖ¶¯ÉèÖÃ X Öá¿Ì¶È£¬Ê¹Æä¸üÕûÆë (ÀýÈçÃ¿ 10 ¸ö Episode Ò»¸ö¿Ì¶È)
plt.xticks(np.arange(0, 110, 30)) 

plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig('Comparison_Reward.png', format='png', dpi=290)
plt.show()

# ==========================================
# 4. »æÍ¼ 2: Design Success Rate Comparison
# ==========================================
plt.figure(figsize=(10, 7.3))
ax = plt.gca()

for fname, label, color in files_config:
    if not os.path.exists(fname): continue
    
    df = pd.read_csv(fname, header=None, on_bad_lines='skip')
    raw_vals = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
    if len(raw_vals) > MAX_STEPS: raw_vals = raw_vals[:MAX_STEPS]
    
    # ¶þÖµ»¯
    success_flags = (raw_vals > -10.0).astype(float)
    
    # °´ Batch ´¦Àí
    n_batches = len(success_flags) // BATCH_SIZE
    success_flags = success_flags[:n_batches * BATCH_SIZE]
    batch_success_rate = success_flags.reshape(-1, BATCH_SIZE).mean(axis=1)
    
    # Æ½»¬
    means, _ = group_average(batch_success_rate, GROUP_SIZE)
    
    # [ÐÞ¸Äµã] ¼ÆËã X Öá×ø±ê (Episodes)
    steps_per_point = GROUP_SIZE * BATCH_SIZE
    total_steps = np.arange(1, len(means) + 1) * steps_per_point
    episodes = total_steps / STEPS_PER_EPISODE
    
    plt.plot(episodes, means, label=label, color=color, linewidth=2.5)

# Ó¦ÓÃÑùÊ½
apply_custom_style(ax)

# [ÐÞ¸Äµã] ×ø±êÖá±êÇ©
plt.xlabel('Episode', fontsize=18)
plt.ylabel('Design Success Rate', fontsize=18)
plt.xticks(np.arange(0, 110, 30)) 

# ÉèÖÃ Y ÖáÎª°Ù·Ö±È¸ñÊ½
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(-0.05, 1.05)

plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig('Comparison_SuccessRate.png', format='png', dpi=290)
plt.show()