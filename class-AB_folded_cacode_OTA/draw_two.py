import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. ºËÐÄ¹¤¾ßº¯Êý (±£ÁôÄãµÄÔ­Ê¼Âß¼­)
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
    ax.tick_params(axis='both', direction='in', labelsize=12, pad=8, length=6, width=3)
    # ÉèÖÃ±ß¿ò´ÖÏ¸£¬Ê¹ÆäÓë tick width Æ¥Åä£¨¿ÉÑ¡£¬ÎªÁËÃÀ¹Û£©
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
            
        # ¼ÆËã Batch Mean (Ã¿7¸öµãËãÒ»¸öÆ½¾ùÖµ£¬´ú±íÒ»²½ÓÅ»¯)
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
    # (ÎÄ¼þÃû, Í¼ÀýÃû, ÑÕÉ«)
    ("results_pure_ppo.csv", "Pure PPO", "blue"), 
    ("results_nsga3.csv",    "NSGA-III",    "green"),
    ("results_ours.csv",     "Ours",        "red"), 
]

MAX_STEPS = 4000
BATCH_SIZE = 7
# GROUP_SIZE ¾ö¶¨ÁËÍ¼µÄÆ½»¬³Ì¶ÈºÍµãµÄÊýÁ¿
# 4000²½ / 7 = 571 ¸ö batch¡£
# ÉèÖÃ group_size=10 ±íÊ¾Ã¿ 70 ´Î·ÂÕæ»­Ò»¸öµã£¬ÇúÏß»á±È½ÏÆ½»¬
GROUP_SIZE = 15 

# ¶ÁÈ¡ËùÓÐÊý¾Ý
data_store = {}
for fname, label, color in files_config:
    data_store[label] = load_batch_data(fname, MAX_STEPS, BATCH_SIZE)

# ==========================================
# 3. »æÍ¼ 1: Average Reward Comparison
# ==========================================
plt.figure(figsize=(10, 6)) # ÉÔÎ¢µ÷Õû³ß´çÒÔÊÊÓ¦ÂÛÎÄ
ax = plt.gca()

for fname, label, color in files_config:
    raw_batches = data_store[label]
    if len(raw_batches) == 0: continue
    
    # Ê¹ÓÃÄãµÄ group_average ´¦ÀíÊý¾Ý
    means, stds = group_average(raw_batches, GROUP_SIZE)
    
    # Éú³É X Öá×ø±ê (Total Simulation Steps)
    # Ã¿¸öµã´ú±í group_size * batch_size ´Î·ÂÕæ
    x_step = GROUP_SIZE * BATCH_SIZE
    steps = np.arange(1, len(means) + 1) * x_step
    
    # »æÍ¼
    plt.plot(steps, means, label=label, color=color, linewidth=2)
    
    # »æÖÆÒõÓ° (Standard Deviation)
    poly = plt.fill_between(steps, means - stds, means + stds,
                            color=color, alpha=0.15) # alpha µ÷Ð¡Ò»µã£¬·ÀÖ¹Èý¸öÖØµþ¿´²»Çå
    poly.set_rasterized(True) # ±£ÁôÄãµÄ¹âÕ¤»¯ÉèÖÃ

# Ó¦ÓÃÄãµÄÑùÊ½
apply_custom_style(ax)

# ×ø±êÖáÉèÖÃ
plt.xlabel('Total Simulation Steps', fontsize=15)
plt.ylabel('Average Reward', fontsize=15)

# ¸ù¾ÝÖ®Ç°µÄ Reward Êý¾Ý·¶Î§ÊÖ¶¯µ÷Õû¿Ì¶È (¿É¸ù¾ÝÊµ¼ÊÇé¿öÐÞ¸Ä)
# plt.xticks(np.arange(0, 4200, 1000))
# plt.yticks(np.arange(-12, 2, 2))

plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig('Comparison_Reward.png', format='png', dpi=290)
plt.show()

# ==========================================
# 4. »æÍ¼ 2: Design Success Rate Comparison
# ==========================================
plt.figure(figsize=(10, 6))
ax = plt.gca()

for fname, label, color in files_config:
    # ÖØÐÂ¶ÁÈ¡Ô­Ê¼Êý¾Ý¼ÆËã³É¹¦ÂÊ
    if not os.path.exists(fname): continue
    
    df = pd.read_csv(fname, header=None, on_bad_lines='skip')
    raw_vals = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
    if len(raw_vals) > MAX_STEPS: raw_vals = raw_vals[:MAX_STEPS]
    
    # ¶þÖµ»¯£º´óÓÚ -10 Ëã³É¹¦
    success_flags = (raw_vals > -10.0).astype(float)
    
    # °´ Batch ´¦Àí (7¸öÀïÓÐ¼¸¸ö³É¹¦£¬ËãÕâ¸ö Batch µÄ³É¹¦ÂÊ)
    n_batches = len(success_flags) // BATCH_SIZE
    success_flags = success_flags[:n_batches * BATCH_SIZE]
    batch_success_rate = success_flags.reshape(-1, BATCH_SIZE).mean(axis=1)
    
    # Ê¹ÓÃ group_average ½øÐÐÆ½»¬
    # ³É¹¦ÂÊÍ¼Í¨³£²»ÐèÒª std ÒõÓ°£¬»òÕß std ÒâÒå²»´ó£¬ÕâÀïÖ»»­¾ùÖµÏß
    means, _ = group_average(batch_success_rate, GROUP_SIZE)
    
    x_step = GROUP_SIZE * BATCH_SIZE
    steps = np.arange(1, len(means) + 1) * x_step
    
    plt.plot(steps, means, label=label, color=color, linewidth=2.5)

# Ó¦ÓÃÄãµÄÑùÊ½
apply_custom_style(ax)

# ×ø±êÖáÉèÖÃ
plt.xlabel('Total Simulation Steps', fontsize=15)
plt.ylabel('Design Success Rate', fontsize=15)

# ÉèÖÃ Y ÖáÎª°Ù·Ö±È¸ñÊ½
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.ylim(-0.05, 1.05) # ÉÔÎ¢Áôµã¿ÕÏ¶

plt.legend(fontsize=12, loc='lower right', frameon=True, framealpha=0.9)
plt.tight_layout()
plt.savefig('Comparison_SuccessRate.png', format='png', dpi=290)
plt.show()