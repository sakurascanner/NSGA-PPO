import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_calculate_success_rate(filename, max_steps=4000, window=100, failure_threshold=-10.0):
    """
    ¶ÁÈ¡ Reward£¬¶þÖµ»¯Îª³É¹¦/Ê§°Ü£¬¼ÆËã»¬¶¯³É¹¦ÂÊ¡£
    """
    if not os.path.exists(filename):
        print(f"¾¯¸æ: ÎÄ¼þ²»´æÔÚ - {filename}")
        return None, None

    try:
        # 1. ¶ÁÈ¡Êý¾Ý
        df = pd.read_csv(filename, header=None, on_bad_lines='skip')
        rewards = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
        
        # 2. ½ØÈ¡Ç° max_steps
        if max_steps is not None and len(rewards) > max_steps:
            rewards = rewards[:max_steps]
            
        # 3. ÅÐ¶¨³É¹¦/Ê§°Ü (Binarization)
        # Reward > -10 ÊÓÎª³É¹¦ (1)£¬·ñÔòÎªÊ§°Ü (0)
        # ÒòÎªÊ§°ÜµÄ Reward ÊÇ -11.0
        success_flags = (rewards > failure_threshold).astype(float)
        
        # 4. ¼ÆËã»¬¶¯³É¹¦ÂÊ (Rolling Success Rate)
        # min_periods=1 ±£Ö¤ÇúÏß´ÓµÚ1²½¾Í¿ªÊ¼»­£¬¶ø²»ÊÇµÈµ½ window ÌîÂú
        success_rate = pd.Series(success_flags).rolling(window=window, min_periods=1).mean()
        
        # 5. Éú³É X Öá
        steps = np.arange(1, len(success_rate) + 1)
        
        return steps, success_rate
        
    except Exception as e:
        print(f"´¦ÀíÎÄ¼þ {filename} Ê±³ö´í: {e}")
        return None, None

# ==========================================
# ÅäÖÃÇøÓò
# ==========================================
files_config = [
    ("reward.csv", "Pure PPO", "blue", "--"), 
    ("nsga3_population_log_0.csv",    "NSGA-III",    "green", ":"),
    ("reward_1w.csv",     "NSGA-enhanced PPO", "red",   "-"), 
]

# ==========================================
# »æÍ¼ÉèÖÃ
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

MAX_STEPS = 4000
ROLLING_WINDOW = 150 
FAILURE_THRESHOLD = -10.0 # ÈÎºÎÐ¡ÓÚ -10 µÄ¶¼±»ÊÓÎª OP ·ÂÕæÊ§°Ü

has_data = False

for filename, label, color, linestyle in files_config:
    steps, success_rate = load_and_calculate_success_rate(
        filename, 
        max_steps=MAX_STEPS, 
        window=ROLLING_WINDOW,
        failure_threshold=FAILURE_THRESHOLD
    )
    
    if steps is not None:
        has_data = True
        # »æÖÆÇúÏß
        ax.plot(steps, success_rate, label=label, color=color, linewidth=2.5, alpha=0.9)

if has_data:
    ax.set_xlabel("Total Simulation Steps", fontsize=12)
    ax.set_ylabel("Success Rate (Rolling Average)", fontsize=12)
    
    # ÉèÖÃ Y ÖáÎª 0% µ½ 100%
    ax.set_ylim(-0.05, 1.05)
    # ¸ñÊ½»¯ Y Öá¿Ì¶ÈÎª°Ù·Ö±È
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    output_filename = "comparison_success_rate.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Í¼±íÒÑ±£´æÎª: {output_filename}")
    plt.show()
else:
    print("Ã»ÓÐÊý¾Ý¿É»æÍ¼")