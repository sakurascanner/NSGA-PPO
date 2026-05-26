import pandas as pd
import numpy as np

# ÄúÌá¹©µÄÕæÊµ Target Öµ
targets = {
    'PSRP_target': -90,
    'PSRN_target': -90,
    'TC_target': 1e-6,
    'Power_target': 200,
    'vos_target': 4e-5,
    'cmrrdc_target': -80,
    'dcgain_target': 130,
    'GBW_target': 1e6,
    'phase_margin_target': 60,
    'sr_target': 4e5,
    'settlingTime_target': 5e-6
}

def process_data(input_file, output_file):
    print(f"ÕýÔÚ¶ÁÈ¡ÎÄ¼þ: {input_file} ...")
    
    # ¶¨ÒåÁÐÃû¡£×¢Òâ£º¼ÙÉèÄúµÄÔ­Êý¾ÝÃ»ÓÐ±íÍ·£¬Èç¹ûÓÐ±íÍ·£¬Çë½« pd.read_csv ÖÐµÄ names ºÍ header ²ÎÊýÈ¥µô¡£
    columns = ['Power', 'dcgain', 'GBW', 'phase_margin', 'TC', 'vos', 'cmrrdc', 'PSRP', 'PSRN', 'sr', 'settlingTime']
    
    try:
        # ¶ÁÈ¡ 8w ÌõÊý¾Ý
        df = pd.read_csv(input_file, names=columns, header=None)
    except FileNotFoundError:
        print(f"´íÎó: ÕÒ²»µ½ÎÄ¼þ {input_file}¡£ÇëÈ·±£¸ÃÎÄ¼þÓë±¾½Å±¾ÔÚÍ¬Ò»Ä¿Â¼ÏÂ¡£")
        return
        
    print(f"³É¹¦¶ÁÈ¡ {len(df)} ÌõÊý¾Ý£¬¿ªÊ¼Ê¹ÓÃÏòÁ¿»¯¼ÓËÙ¼ÆËã reward ...")

    # ==========================
    # ÏòÁ¿»¯²Ù×÷ (¼«ËÙ´¦Àí)
    # ==========================
    
    # 1. »ù´¡µÃ·Ö¼ÆËãº¯Êý: max(min((target - val) / (target + val), 0), -1)
    def calc_standard(val, target):
        return np.clip((target - val) / (target + val), -1, 0)
        
    df['TC_score'] = calc_standard(df['TC'], targets['TC_target'])
    df['Power_score'] = calc_standard(df['Power'], targets['Power_target'])
    df['vos_score'] = calc_standard(df['vos'], targets['vos_target'])
    df['settlingTime_score'] = calc_standard(df['settlingTime'], targets['settlingTime_target'])
    
    # 2. SR Score ¼ÆËã: (val - target) / (val + target)
    df['sr_score'] = np.clip((df['sr'] - targets['sr_target']) / (df['sr'] + targets['sr_target']), -1, 0)
    
    # 3. CMRR, PSRP, PSRN ¼ÆËãÂß¼­
    def calc_db_score(val, target):
        score = np.clip((val - target) / (val + target), -1, 0)
        score = np.where(val < target, 0, score) # Èç¹û val < target, ÔòÉèÎª 0
        return np.where(val > 0, -1, score)      # Èç¹û val > 0, ÔòÉèÎª -1

    df['cmrrdc_score'] = calc_db_score(df['cmrrdc'], targets['cmrrdc_target'])
    df['PSRP_score'] = calc_db_score(df['PSRP'], targets['PSRP_target'])
    df['PSRN_score'] = calc_db_score(df['PSRN'], targets['PSRN_target'])

    # 4. DC Gain, GBW, Phase Margin ¼ÆËãÂß¼­
    df['dcgain_score'] = np.clip((df['dcgain'] - targets['dcgain_target']) / (df['dcgain'] + targets['dcgain_target']), -1, 0)
    df['GBW_score'] = np.clip((df['GBW'] - targets['GBW_target']) / (df['GBW'] + targets['GBW_target']), -1, 0)
    df['phase_margin_score'] = np.clip((df['phase_margin'] - targets['phase_margin_target']) / (df['phase_margin'] + targets['phase_margin_target']), -1, 0)
    
    # ÀûÓÃ²¼¶ûÑÚÂëÅúÁ¿´¦Àí dcgain <= 0 µÄÇé¿ö
    invalid_dc_mask = df['dcgain'] <= 0
    df.loc[invalid_dc_mask, ['dcgain_score', 'GBW_score', 'phase_margin_score']] = -1
    
    # ==========================
    # »ã×Ü Reward
    # ==========================
    df['reward'] = (df['TC_score'] + df['Power_score'] + df['vos_score'] + 
                    df['cmrrdc_score'] + df['PSRP_score'] + df['PSRN_score'] + 
                    df['dcgain_score'] + df['GBW_score'] + df['phase_margin_score'] + 
                    df['sr_score'] + df['settlingTime_score'])
                    
    # ==========================
    # ÌáÈ¡ÐèÒªµÄÁÐ²¢±£´æ
    # ==========================
    output_df = df[['reward'] + columns + ['reward']]
    output_df.to_csv(output_file, index=False)
    print(f"¼ÆËãÍê³É£¡°üº¬ reward µÄÊý¾ÝÒÑ±£´æÖÁ {output_file}")

if __name__ == '__main__':
    # µ÷ÓÃº¯Êý´¦ÀíÊý¾Ý£ºÊäÈëÎÄ¼þÎª gen.csv£¬Êä³öÎÄ¼þÎª gen_with_reward.csv
    process_data('gen.csv', 'gen_with_reward.csv')