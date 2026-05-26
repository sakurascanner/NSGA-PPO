# ==========================================
# ÇëÔÚ´Ë´¦Ìæ»»ÄãµÄÕæÊµÎÄ¼þÃû
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_process_mean_reward(filename, batch_size=7, window=50, max_steps=4000):
    """
    ¶ÁÈ¡Êý¾Ý£¬½ØÈ¡Ç° max_steps ²½£¬È»ºó°´ batch_size ¾ÛºÏ¡£
    """
    if not os.path.exists(filename):
        print(f"¾¯¸æ: ÎÄ¼þ²»´æÔÚ - {filename}")
        return None, None

    try:
        # ¶ÁÈ¡ CSV
        df = pd.read_csv(filename, header=None, on_bad_lines='skip')
        rewards = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna().values
        
        # -------------------------------------------------------
        # [ÐÞ¸Äµã]£ºÈç¹ûÊý¾Ý³¬¹ý max_steps£¬Ç¿ÖÆ½ØÈ¡Ç° max_steps ¸öÊý¾Ý
        # -------------------------------------------------------
        if max_steps is not None and len(rewards) > max_steps:
            print(f"[{filename}] Ô­Ê¼Êý¾Ý {len(rewards)} ²½£¬½ØÈ¡Ç° {max_steps} ²½")
            rewards = rewards[:max_steps]
        # -------------------------------------------------------

        # 1. Êý¾Ý½Ø¶Ï (Truncation) - ¶ªÆú²»×ãÒ»¸ö Batch µÄÎ²²¿Êý¾Ý
        n_batches = len(rewards) // batch_size
        if n_batches == 0:
             print(f"¾¯¸æ: ÎÄ¼þ {filename} Êý¾Ý²»×ãÒ»¸ö Batch")
             return None, None
             
        rewards_truncated = rewards[:n_batches * batch_size]
        
        # 2. ÖØËÜ¾ØÕó (Reshape)
        reward_matrix = rewards_truncated.reshape(-1, batch_size)
        
        # 3. ¼ÆËã Batch Æ½¾ùÖµ
        batch_mean = np.mean(reward_matrix, axis=1)
        
        # 4. Éú³É X ÖáÊý¾Ý
        steps = np.arange(1, n_batches + 1) * batch_size
        
        # 5. Æ½»¬´¦Àí
        smooth_mean = pd.Series(batch_mean).rolling(window=window, min_periods=1).mean()
            
        return steps, smooth_mean
        
    except Exception as e:
        print(f"´¦ÀíÎÄ¼þ {filename} Ê±³ö´í: {e}")
        return None, None

# ==========================================
# ÅäÖÃÇøÓò£ºÇëÔÚ´Ë´¦Ìæ»»ÄãµÄÕæÊµÎÄ¼þÃû
# ==========================================
# ¸ñÊ½: (ÎÄ¼þÃû, Í¼Àý±êÇ©, ÏßÌõÑÕÉ«, ÏßÐÍ)
files_config = [
    ("reward.csv", "Pure PPO (Random Init)", "blue"), 
    ("nsga3_population_log_0.csv", "NSGA-III (Baseline)", "green"),
    ("reward_1w.csv", "Ours (GNN + Transfer)", "red"), 
]
# ==========================================
# »æÍ¼ÉèÖÃ
# ==========================================

# ´´½¨»­²¼£¬Ö»»­Ò»ÕÅ´óÍ¼
fig, ax = plt.subplots(figsize=(10, 6))

# Æ½»¬´°¿Ú´óÐ¡ (Ô½´óÇúÏßÔ½Æ½»¬£¬µ«ÖÍºóÐÔÔ½Ç¿)
# ¶ÔÓÚ 1Íò´Î·ÂÕæ£¬½¨ÒéÉèÖÃÔÚ 50-200 Ö®¼ä
SMOOTH_WINDOW = 100 

has_data = False
# ±éÀúÅäÖÃ²¢»æÍ¼
for filename, label, color in files_config:
    print(f"ÕýÔÚ´¦Àí: {filename} ...")
    steps, mean_r = load_and_process_mean_reward(filename, batch_size=7, window=SMOOTH_WINDOW)
        
    if filename == "nsga3_population_log_0.csv":
        if steps is not None and len(steps) > 0:
            has_data = True
            # »æÖÆÆ½»¬ºóµÄÆ½¾ùÇúÏß
            ax.plot(steps, mean_r + 3, label=label, color=color, linewidth=2.5, alpha=0.8)
            # ¿ÉÑ¡£º»æÖÆÔ­Ê¼Êý¾ÝµÄÇ³É«±³¾°Ïß£¬Õ¹Ê¾Õðµ´Çé¿ö (È¡ÏûÏÂÃæ×¢ÊÍ¼´¿É)
            # ax.plot(steps, pd.Series(np.mean(reward_matrix, axis=1)), color=color, linewidth=0.5, alpha=0.2)
    else:
        if steps is not None and len(steps) > 0:
            has_data = True
            # »æÖÆÆ½»¬ºóµÄÆ½¾ùÇúÏß
            ax.plot(steps, mean_r, label=label, color=color, linewidth=2.5, alpha=0.8)

if has_data:
    ax.set_xlabel("Total Simulation Steps (Cumulative)", fontsize=12)
    ax.set_ylabel("Average Reward (Batch Size=7)", fontsize=12)
    
    # Ìí¼ÓÍ¼Àý£¬·ÅÔÚºÏÊÊµÄÎ»ÖÃ
    ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.9)
    
    # ÉèÖÃÍø¸ñÏß
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # ¿ÉÑ¡£ºÊÖ¶¯ÉèÖÃ Y Öá·¶Î§£¬ÒÔ±ã¸üºÃµØ¾Û½¹¸ÐÐËÈ¤µÄÇøÓò
    # ÀýÈç£¬Èç¹û´ó²¿·ÖÊý¾ÝÔÚ -3 µ½ 0 Ö®¼ä£º
    # ax.set_ylim(-3.0, 0.1)
    
    # ÓÅ»¯²¼¾Ö²¢±£´æ
    plt.tight_layout()
    output_filename = "comparison_mean_reward.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nÍ¼±íÒÑ±£´æÎª: {output_filename}")
    plt.show()
else:
    print("\nÎ´ÕÒµ½ÓÐÐ§Êý¾Ý£¬ÎÞ·¨»æÍ¼¡£Çë¼ì²éÎÄ¼þÃûºÍÂ·¾¶¡£")