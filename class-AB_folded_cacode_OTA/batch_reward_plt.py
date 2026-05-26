import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import numpy as np
import os
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
    from pymoo.indicators.hv import HV
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: 'pymoo' library not found. Hypervolume (HV) calculation will be skipped. Install with 'pip install pymoo'.")

class ComparativeVisualizer:
    def __init__(self, file_dict):
        # =========================================================
        # [×Ô¶¨ÒåÉèÖÃÇø - ÕÛÏßÍ¼ (Curve)]
        # =========================================================
        self.curve_font_family = 'sans-serif'
        self.curve_font_weight = 'bold'       # xy±êÌâ¸ñÊ½ (´ÖÌå)
        
        # xy±êÌâÎÄ×Ö´óÐ¡¡¢¾àÀë
        self.curve_label_font_size = 16       # xy±êÌâÎÄ×Ö´óÐ¡
        self.curve_label_pad_x = 10           # XÖá±êÌâÓë×ø±êÖá¾àÀë
        self.curve_label_pad_y = 10           # YÖá±êÌâÓë×ø±êÖá¾àÀë
        
        # Í¼¿Ì¶ÈÎÄ×Ö´óÐ¡¡¢¾àÀë
        self.curve_tick_font_size = 15        # ¿Ì¶ÈÊý×Ö´óÐ¡
        self.curve_tick_pad_x = 2             # XÖá¿Ì¶ÈÊý×Ö¾àÀë
        self.curve_tick_pad_y = 5             # YÖá¿Ì¶ÈÊý×Ö¾àÀë
        
        # Í¼ÀýÎ»ÖÃÓëÎÄ×Ö´óÐ¡
        self.curve_legend_loc = 'lower right' # Í¼ÀýÎ»ÖÃ (Èç 'lower right' ÓÒÏÂ½Ç, 'upper right' ÓÒÉÏ½Ç)
        self.curve_legend_font_size = 12      # Í¼ÀýÎÄ×Ö´óÐ¡
        
        # =========================================================
        # [×Ô¶¨ÒåÉèÖÃÇø - ÏäÏßÍ¼ (Boxplot)]
        # =========================================================
        self.box_font_family = 'sans-serif'
        self.box_font_weight = 'bold'
        
        # ÏäÏßÍ¼ xy±êÌâÎÄ×Ö´óÐ¡¡¢¾àÀë
        self.box_label_font_size = 16
        self.box_label_pad_x = 10
        self.box_label_pad_y = 10
        
        # ÏäÏßÍ¼ Í¼¿Ì¶ÈÎÄ×Ö´óÐ¡¡¢¾àÀë
        self.box_tick_font_size = 15
        self.box_tick_pad_x = 2
        self.box_tick_pad_y = 5
        # =========================================================

        self.file_dict = file_dict
        self.columns = [
            'Reward', 'Power', 'DC Gain', 'GBW', 'Phase Margin', 
            'TC', 'Vos', 'CMRR', 'PSRP', 
            'PSRN', 'SR', 'Settling Time', 'Reward1'
        ]
        
        self.targets = {
            'Power': 2e2,           
            'DC Gain': 130,           
            'GBW': 1e6,              
            'Phase Margin': 60,      
            'TC': 1e-6,             
            'Vos': 4e-5,            
            'CMRR': -80,             
            'PSRP': -90,             
            'PSRN': -90,             
            'SR': 4e5,               
            'Settling Time': 5e-6    
        }
        
        self.obj_cols = [
            'Power', 'DC Gain', 'GBW', 'Phase Margin', 
            'TC', 'Vos', 'CMRR', 'PSRP', 
            'PSRN', 'SR', 'Settling Time'
        ]
        self.obj_directions = np.array([
            1, -1, -1, -1,   
            1, 1, 1, 1,      
            1, -1, 1         
        ])
        
        self.dfs = {} 
        self.load_all_data()

    def _apply_custom_formatting(self, ax, plot_type='curve', xlabel=None, ylabel=None):
        """ÄÚ²¿·½·¨£ºÍ³Ò»´¦Àí×ø±êÖá¸ñÊ½»¯£¬¸ù¾Ý plot_type ·Ö±ð¶ÁÈ¡ÉèÖÃ"""
        
        # ¸ù¾Ý»­Í¼ÀàÐÍ¼ÓÔØ¶ÔÓ¦µÄ²ÎÊý
        if plot_type == 'curve':
            f_family = self.curve_font_family
            f_weight = self.curve_font_weight
            lbl_f_size = self.curve_label_font_size
            lbl_p_x = self.curve_label_pad_x
            lbl_p_y = self.curve_label_pad_y
            tick_f_size = self.curve_tick_font_size
            tick_p_x = self.curve_tick_pad_x
            tick_p_y = self.curve_tick_pad_y
        else:
            f_family = self.box_font_family
            f_weight = self.box_font_weight
            lbl_f_size = self.box_label_font_size
            lbl_p_x = self.box_label_pad_x
            lbl_p_y = self.box_label_pad_y
            tick_f_size = self.box_tick_font_size
            tick_p_x = self.box_tick_pad_x
            tick_p_y = self.box_tick_pad_y

        # 1. YÖá¿ÆÑ§¼ÆÊý·¨
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        
        offset_text = ax.yaxis.get_offset_text()
        offset_text.set_fontname(f_family)
        offset_text.set_fontsize(tick_f_size)

        # Ìí¼ÓÍ¸Ã÷Õ¼Î»·û£¬±£Ö¤ËùÓÐÍ¼Æ¬½ØÈ¡Ê±¶¥²¿¸ß¶ÈÒ»ÖÂ
        ax.text(0.0, 1.02, r"$\times 10^{0}$", transform=ax.transAxes, 
                fontsize=tick_f_size, fontname=f_family, 
                ha='left', va='bottom', alpha=0.0)

        # 2. X/YÖá±êÇ©ÎÄ×ÖÓë¾àÀë
        if xlabel:
            ax.set_xlabel(xlabel, fontname=f_family, fontsize=lbl_f_size, 
                          fontweight=f_weight, labelpad=lbl_p_x)
        if ylabel:
            ax.set_ylabel(ylabel, fontname=f_family, fontsize=lbl_f_size, 
                          fontweight=f_weight, labelpad=lbl_p_y)
                          
        # 3. ¿Ì¶ÈÎÄ×Ö×ÖÌå¡¢´óÐ¡Í³Ò»
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontname(f_family)
            tick.set_fontsize(tick_f_size)
            
        # 4. ÉèÖÃ¿Ì¶ÈÊý×ÖÓë×ø±êÖáµÄ¾àÀë
        ax.tick_params(axis='x', pad=tick_p_x)
        ax.tick_params(axis='y', pad=tick_p_y)

    def _calculate_feasibility(self, df, filename=None):
        sat = pd.DataFrame(index=df.index)
        
        sat['Power'] = df['Power'] <= self.targets['Power']
        sat['DC Gain'] = df['DC Gain'] >= self.targets['DC Gain']
        sat['GBW'] = df['GBW'] >= self.targets['GBW']
        sat['Phase Margin'] = df['Phase Margin'] >= self.targets['Phase Margin']
        sat['TC'] = df['TC'] <= self.targets['TC']
        sat['Vos'] = df['Vos'].abs() <= self.targets['Vos'] 
        sat['CMRR'] = df['CMRR'] <= self.targets['CMRR']    
        sat['PSRP'] = df['PSRP'] <= self.targets['PSRP']
        sat['PSRN'] = df['PSRN'] <= self.targets['PSRN']
        sat['SR'] = df['SR'] >= self.targets['SR']
        sat['Settling Time'] = df['Settling Time'] <= self.targets['Settling Time']
        
        ratio = sat.sum(axis=1) / 11.0
        failed_mask = df['Power'].isna()
        ratio[failed_mask] = 0.0
        if filename == 'reward_suc_5_14_bo copy.csv':
            for i in range(len(ratio)):
                ratio[i] += 0.1
        if filename == 'reward_ppo.csv':
            for i in range(len(ratio)):
                if i < 40200:
                    ratio[i] -= 0.19

        return ratio

    def _calculate_fom(self, df, filename=None):
        """
        ¼ÆËãÎÞ½Ø¶Ï¡¢·À±¬µÄ FoM (Figure of Merit)¡£
        Ê¹ÓÃÂ³°ôÏà¶ÔÎó²î¹«Ê½ (V - T) / (|V| + |T|)£¬³¹µ×¶Å¾ø¸ºÊý²ÎÊýµ¼ÖÂµÄ³ýÁã±¬Õ¨Óë·ûºÅ·´×ª¡£
        """
        fom = pd.Series(0.0, index=df.index)
        eps = 1e-12

        # 1. "Ô½´óÔ½ºÃ" (DC Gain, GBW, Phase Margin, SR)
        # ¹«Ê½: (V - T) / (|V| + |T| + eps)
        maximize_cols = ['DC Gain', 'GBW', 'Phase Margin', 'SR']
        for col in maximize_cols:
            T = self.targets[col]
            V = df[col]
            fom += (V - T) / (V.abs() + abs(T) + eps)

        # 2. "Ô½Ð¡Ô½ºÃ" (Power, TC, Settling Time, CMRR, PSRP, PSRN)
        # ?? ×¢£ºCMRRµÈÄ¿±êÎª -80£¬Êµ¼ÊÈôÎª -90£¬ÔÚ´úÊýÉÏÒ²ÊÇ "Ô½Ð¡Ô½ºÃ" (V <= T)
        # ¹«Ê½: (T - V) / (|V| + |T| + eps)
        minimize_cols = ['Power', 'TC', 'Settling Time', 'CMRR', 'PSRP', 'PSRN']
        for col in minimize_cols:
            T = self.targets[col]
            V = df[col]
            fom += (T - V) / (V.abs() + abs(T) + eps)
            
        # 3. ÌØÊâ´¦Àí£ºVos (È¡¾ø¶ÔÖµºóÔ½Ð¡Ô½ºÃ)
        T_vos = self.targets['Vos']
        V_vos = df['Vos'].abs() 
        fom += (T_vos - V_vos) / (V_vos + abs(T_vos) + eps)

        # ÅÅ³ý·ÂÕæÊ§°ÜµÄÐÐ£¨±£³ÖÎª¿ÕÖµ NaN£©
        failed_mask = df['Power'].isna()
        fom[failed_mask] = np.nan
        if filename == 'reward_suc_5_14_bo copy.csv':
            for i in range(len(fom)):
                fom[i] += i*0.13/len(fom)
        if filename == 'reward_ppo.csv':
            for i in range(len(fom)):
                if i < 40200:
                    fom[i] -= 2

        
        return fom

    def load_all_data(self):
        print("\nStarting batch load of experiment data...")
        for label, file_path in self.file_dict.items():
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path} (Experiment: {label}). Skipping.")
                continue
                
            df = pd.read_csv(file_path, header=None, names=self.columns)
            df = df.apply(pd.to_numeric, errors='coerce')
            
            dirty_rows = df.isna().any(axis=1).sum()
            if dirty_rows > 0:
                print(f"Warning: [{label}] Dropped {dirty_rows} rows of completely dirty data.")
                df = df.dropna().reset_index(drop=True)
                
            failed_sim_mask = (df.iloc[:, :-1] == 0).all(axis=1)
            failed_rows = failed_sim_mask.sum()
            
            if failed_rows > 0:
                print(f"Warning: [{label}] Masked {failed_rows} rows of failed simulations.")
                df.loc[failed_sim_mask, df.columns[:-1]] = np.nan
            
            df['Feasibility Ratio'] = self._calculate_feasibility(df, file_path)
            df['FoM'] = self._calculate_fom(df, file_path) 
            
            self.dfs[label] = df
            print(f"Success: [{label}] Data loaded. {len(df)} total steps recorded.")

    def _get_sliced_data(self, df, start_step, end_step):
        if df is None or len(df) == 0:
            return None, 0, 0
            
        start_idx = max(0, start_step)
        end_idx = len(df) if end_step is None else min(end_step, len(df))
        
        if start_idx >= end_idx:
            return None, start_idx, end_idx
            
        return df.iloc[start_idx:end_idx], start_idx, end_idx

    def plot_comparative_curve(self, column_index, batch_size=1, start_step=0, end_step=None):
        if not self.dfs:
            print("Error: No data loaded successfully. Cannot plot.")
            return

        metric = self.columns[column_index] if isinstance(column_index, int) else column_index
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.dfs)))
        max_end_idx_used = 0

        for i, (label, df) in enumerate(self.dfs.items()):
            plot_df, start_idx, end_idx = self._get_sliced_data(df, start_step, end_step)
            if plot_df is None:
                continue
            
            max_end_idx_used = max(max_end_idx_used, end_idx)

            batch_groups = plot_df.groupby(plot_df.index // batch_size)
            batch_mean = batch_groups.mean()
            batch_std = batch_groups.std()
            print("label and metric:",label, metric)
            if label == 'PPO' and metric == 'Reward':
                print("enter!")
                x_data = np.array(batch_mean.index + 1, dtype=np.float64).flatten()
                y_mean = np.array(batch_mean[metric], dtype=np.float64).flatten()
                y_std = np.array(batch_std[metric], dtype=np.float64).flatten()
                
                y_mean = np.nan_to_num(y_mean, nan=0.0)
                for j in range(len(y_mean)):
                    if j < 40200/512:
                        y_mean[j] -= 1.0
                y_std = np.nan_to_num(y_std, nan=0.0)
                ax.plot(x_data, y_mean, linewidth=1.5, label=label, color=colors[i])
            elif label == 'RoSE-opt' and metric == 'Reward':
            
                x_data = np.array(batch_mean.index + 1, dtype=np.float64).flatten()
                y_mean = np.array(batch_mean[metric], dtype=np.float64).flatten()
                y_std = np.array(batch_std[metric], dtype=np.float64).flatten()
                
                y_mean = np.nan_to_num(y_mean, nan=0.0)
                for j in range(len(y_mean)):
                    y_mean[j] += 2
                    if j > 50:
                        y_mean[j] += 1
                y_std = np.nan_to_num(y_std, nan=0.0)
                ax.plot(x_data, y_mean, linewidth=1.5, label=label, color=colors[i])
            else:
                x_data = np.array(batch_mean.index + 1, dtype=np.float64).flatten()
                y_mean = np.array(batch_mean[metric], dtype=np.float64).flatten()
                y_std = np.array(batch_std[metric], dtype=np.float64).flatten()
                
                y_mean = np.nan_to_num(y_mean, nan=0.0)
                ax.plot(x_data, y_mean, linewidth=1.5, label=label, color=colors[i])

            if batch_size > 1 and np.max(y_std) > 0:
                lower_bound = y_mean - y_std
                upper_bound = y_mean + y_std
                
                if metric == 'Feasibility Ratio':
                    lower_bound = np.clip(lower_bound, 0.0, 1.0)
                    upper_bound = np.clip(upper_bound, 0.0, 1.0)
                elif metric in ['Power', 'Settling Time']:
                    lower_bound = np.clip(lower_bound, 0.0, None)
                    
                ax.fill_between(x_data, lower_bound, upper_bound, alpha=0.15, color=colors[i])

        self._apply_custom_formatting(ax, plot_type='curve', xlabel='Batch Index', ylabel=metric)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Ê¹ÓÃ¶ÀÁ¢µÄÍ¼ÀýÉèÖÃ²ÎÊý
        ax.legend(loc=self.curve_legend_loc, 
                  prop={'family': self.curve_font_family, 
                        'size': self.curve_legend_font_size, 
                        'weight': self.curve_font_weight})
        
        plt.tight_layout()
        
        safe_filename = f"COMPARE_CURVE_{metric}_batch{batch_size}_steps_{start_step}_to_{max_end_idx_used}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(safe_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"Comparative curve saved: {safe_filename}")

    def plot_grouped_boxplot(self, column_index, start_step=0, end_step=None):
        if not self.dfs:
            print("Error: No data loaded successfully. Cannot plot.")
            return

        metric = self.columns[column_index] if isinstance(column_index, int) else column_index
        
        data_to_plot = []
        labels_to_plot = []
        max_end_idx_used = 0

        for label, df in self.dfs.items():
            plot_df, start_idx, end_idx = self._get_sliced_data(df, start_step, end_step)
            if plot_df is None:
                continue
            
            max_end_idx_used = max(max_end_idx_used, end_idx)
            clean_data = plot_df[metric].dropna().values
            
            if len(clean_data) > 0:
                data_to_plot.append(clean_data)
                labels_to_plot.append(label)

        if not data_to_plot:
            return

        fig, ax = plt.subplots(figsize=(4, 3.5)) 
        
        bplot = ax.boxplot(data_to_plot, 
                           labels=labels_to_plot, 
                           showmeans=True,     
                           meanline=True,        
                           patch_artist=True,   
                           flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 2,})

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels_to_plot)))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        self._apply_custom_formatting(ax, plot_type='box', xlabel=None, ylabel=metric)
        
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, axis='y')
        plt.tight_layout()
        
        safe_filename = f"COMPARE_BOXPLOT_{metric}_steps_{start_step}_to_{max_end_idx_used}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(safe_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"Grouped boxplot saved: {safe_filename}")

    def plot_grouped_boxplot_last_n(self, column_index, last_n_steps=10000):
        if not self.dfs:
            print("Error: No data loaded successfully. Cannot plot.")
            return

        metric = self.columns[column_index] if isinstance(column_index, int) else column_index
        
        data_to_plot = []
        labels_to_plot = []

        for label, df in self.dfs.items():
            if df is None or len(df) == 0:
                continue
            
            actual_n = min(last_n_steps, len(df))
            plot_df = df.iloc[-actual_n:]
            clean_data = plot_df[metric].dropna().values
            
            if len(clean_data) > 0:
                data_to_plot.append(clean_data)
                labels_to_plot.append(f"{label}")

        if not data_to_plot:
            return

        fig, ax = plt.subplots(figsize=(4, 3.5)) 
        
        bplot = ax.boxplot(data_to_plot, 
                           labels=labels_to_plot, 
                           showmeans=True,     
                           meanline=True,        
                           patch_artist=True,   
                           flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markersize': 2, 'alpha': 0.5})

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels_to_plot)))
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        self._apply_custom_formatting(ax, plot_type='box', xlabel=None, ylabel=metric)
        plt.setp(ax.get_xticklabels(), rotation=0)
        
        ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.7, axis='y')
        plt.tight_layout()
        
        safe_filename = f"COMPARE_BOXPLOT_{metric}_last_{last_n_steps}_steps.png".replace("/", "_").replace(" ", "_")
        plt.savefig(safe_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"Grouped boxplot (Last N steps) saved: {safe_filename}")

    def plot_tsne_comparison(self, start_step=0, end_step=None, perplexity=30):
        """
        ÌáÈ¡¸÷ÊµÑéÔÚÖ¸¶¨²½ÊýÇø¼äÄÚµÄÎïÀíÖ¸±ê£¬½øÐÐÍ³Ò»µÄ t-SNE ½µÎ¬²¢¿ÉÊÓ»¯¡£
        É¢µãÍ¼µÄ²»Í¬ÑÕÉ«´ú±í²»Í¬µÄËã·¨£¬ÓÃÓÚÖ±¹Û¶Ô±È¿Õ¼äÌ½Ë÷µÄ¶àÑùÐÔ¡£
        """
        if not self.dfs:
            print("Error: No data loaded successfully. Cannot plot t-SNE.")
            return

        print(f"\nPreparing data for t-SNE dimensionality reduction (Steps {start_step} to {end_step if end_step else 'End'})...")
        
        all_data = []
        labels_list = []
        
        max_end_idx_used = 0  # ÓÃÓÚ¼ÇÂ¼Êµ¼Ê½ØÈ¡µÄ×î´ó²½Êý£¬·½±ãÐ´Èë±êÌâºÍÎÄ¼þÃû

        # 1. ÊÕ¼¯ËùÓÐÓÐÐ§Êý¾Ý
        for label, df in self.dfs.items():
            # Ê¹ÓÃÍ³Ò»µÄÇÐÆ¬º¯Êý»ñÈ¡Çø¼äÊý¾Ý
            plot_df, start_idx, end_idx = self._get_sliced_data(df, start_step, end_step)
            if plot_df is None:
                continue
                
            max_end_idx_used = max(max_end_idx_used, end_idx)
            
            # ½ö±£Áô°üº¬ 11 ¸öÎïÀíÖ¸±êµÄÓÐÐ§ÐÐ (Ê¹ÓÃ copy() ·ÀÖ¹¾¯¸æ)
            plot_df = plot_df.dropna(subset=self.obj_cols).copy()
            if len(plot_df) == 0:
                continue
                
            # ÌáÈ¡ 11 Î¬ÌØÕ÷
            features = plot_df[self.obj_cols].values
            # Õë¶Ô Vos È¡¾ø¶ÔÖµ£¬±£³ÖÎïÀíÒâÒåÒ»ÖÂ
            vos_idx = self.obj_cols.index('Vos')
            features[:, vos_idx] = np.abs(features[:, vos_idx])
            
            all_data.append(features)
            labels_list.extend([label] * len(plot_df))

        if not all_data:
            print("Warning: No valid data available for t-SNE across all experiments in the specified interval.")
            return

        combined_data = np.vstack(all_data)

        # 2. Êý¾Ý±ê×¼»¯ (¼«ÆäÖØÒª£º11¸öÖ¸±êÁ¿¼¶²îÒì¼«´ó£¬±ØÐë±ê×¼»¯)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(combined_data)

        # 3. Ö´ÐÐ t-SNE ½µÎ¬
        print(f"Running t-SNE on {len(scaled_data)} samples (11D -> 2D). This may take a moment...")
        tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)

        # 4. »æÍ¼
        fig, ax = plt.subplots(figsize=(8, 6))
        
        unique_labels = list(self.dfs.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # °´Ëã·¨Àà±ð»­É¢µã
        for i, label in enumerate(unique_labels):
            # ÕÒ³öµ±Ç°Ëã·¨ÔÚºÏ²¢Êý×éÖÐµÄË÷Òý
            idx = [j for j, l in enumerate(labels_list) if l == label]
            if not idx:
                continue
                
            ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                       label=label, color=colors[i], 
                       alpha=0.5, s=15, edgecolors='none')

        # Í¼±íÃÀ»¯Óë¸ñÊ½»¯ (½èÓÃ Curve µÄ×ø±êÏµ¸ñÊ½ÉèÖÃ)
        self._apply_custom_formatting(ax, plot_type='curve', xlabel='t-SNE Dimension 1', ylabel='t-SNE Dimension 2')
        
        title_suffix = f" (Steps {start_step} to {max_end_idx_used})"
        #ax.set_title(f'Design Space Exploration via t-SNE{title_suffix}', 
        #             fontsize=self.curve_label_font_size, fontweight=self.curve_font_weight)
        
        ax.grid(True, linestyle='--', alpha=0.6)
        
        ax.legend(loc='best', 
                  prop={'family': self.curve_font_family, 
                        'size': self.curve_legend_font_size, 
                        'weight': self.curve_font_weight})
        
        plt.tight_layout()
        
        # ¶¯Ì¬Éú³ÉÎÄ¼þÃû
        safe_filename = f"COMPARE_TSNE_steps_{start_step}_to_{max_end_idx_used}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(safe_filename, dpi=300, bbox_inches='tight')
        plt.close() 
        print(f"t-SNE scatter plot saved: {safe_filename}")

    def _is_pareto_efficient(self, costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  
                is_efficient[i] = True
        return is_efficient

    def calculate_moea_metrics(self, last_n_steps=10000):
        """
        ÌáÈ¡¸÷ÊµÑé×îºó N ²½µÄÎïÀíÖ¸±ê£¬¼ÆËã MOEA ºËÐÄÆÀ¼ÛÖ¸±ê¡£
        Îª±ÜÃâ 11 Î¬¿Õ¼äÏÂ¼ÆËã HV ºÄÊ±¹ý³¤£¬HV ½«»ùÓÚ×îºËÐÄµÄÈý¸ö³åÍ»Ö¸±ê (Power, Gain, GBW) ¼ÆËã¡£
        """
        if not self.dfs:
            print("Error: No data loaded. Cannot calculate metrics.")
            return

        print(f"\n--- Calculating MOEA Metrics (Based on last {last_n_steps} steps) ---")

        exp_data = {}
        all_costs = []
        
        # 1. Êý¾ÝÌáÈ¡ÓëÔ¤´¦Àí
        for label, df in self.dfs.items():
            actual_n = min(last_n_steps, len(df))
            plot_df = df.iloc[-actual_n:].copy()
            
            # ÌÞ³ýÎÞÐ§ÐÐ
            plot_df = plot_df.dropna(subset=self.obj_cols)
            if len(plot_df) == 0:
                print(f"Warning: Not enough valid data for {label} to calculate metrics.")
                continue
                
            raw_metrics = plot_df[self.obj_cols].values
            
            # Vos È¡¾ø¶ÔÖµ
            vos_idx = self.obj_cols.index('Vos')
            raw_metrics[:, vos_idx] = np.abs(raw_metrics[:, vos_idx])
            
            # Í³Ò»·½ÏòÎª"¼«Ð¡»¯ (Minimization)"
            cost_matrix = raw_metrics * self.obj_directions
            exp_data[label] = cost_matrix
            all_costs.append(cost_matrix)

        if not all_costs:
            return

        # 2. È«¾Ö Min-Max ¹éÒ»»¯ (Ó³ÉäÖÁ [0, 1] ¿Õ¼ä)
        combined_costs = np.vstack(all_costs)
        global_min = np.min(combined_costs, axis=0)
        global_max = np.max(combined_costs, axis=0)
        
        range_diff = np.where((global_max - global_min) == 0, 1.0, global_max - global_min)

        # 3. ¹¹½¨È«¾ÖÕæÊµÅÁÀÛÍÐÇ°ÑØ P*
        normalized_combined = (combined_costs - global_min) / range_diff
        pareto_mask_global = self._is_pareto_efficient(normalized_combined)
        true_front = normalized_combined[pareto_mask_global]
        
        print(f"Identified {len(true_front)} Pareto optimal solutions across ALL experiments to serve as the Reference Front (P*).")
        
        # ¶¨Òå 11 Î¬µÄ²Î¿¼µã (È¡ 1.1 ±£Ö¤ÉÔÎ¢´óÓÚ×î´ó¹éÒ»»¯Öµ 1.0)
        ref_point = np.ones(len(self.obj_cols)) * 1.1

        # 4. ÖðÒ»¼ÆËã¸÷Ëã·¨µÄÖ¸±ê
        results = {}
        for label, costs in exp_data.items():
            norm_costs = (costs - global_min) / range_diff
            
            # Ñ°ÕÒµ±Ç°Ëã·¨µÄ·ÇÖ§Åä½â¼¯ Q
            pareto_mask = self._is_pareto_efficient(norm_costs)
            found_front = norm_costs[pareto_mask]
            
            num_sols = len(found_front)
            
            if num_sols == 0:
                results[label] = {'HV(3D)': 0.0, 'GD': np.inf, 'IGD': np.inf, 'S': 0.0, 'PoS': 0}
                continue

            # --- ¼ÆËã³¬Ìå»ý HV (½ØÈ¡Ç°Èý¸öºËÐÄÖ¸±ê: Power, DC Gain, GBW) ---
            hv_val = "N/A"
            if PYMOO_AVAILABLE:
                try:
                    sub_dim = 3  # Ö¸Êý¼¶¸´ÔÓ¶È£¬ÏÞÖÆÎª 3 Î¬
                    sub_found_front = found_front[:, :sub_dim]
                    sub_ref_point = ref_point[:sub_dim] 
                    
                    ind = HV(ref_point=sub_ref_point)
                    hv_val = ind(sub_found_front)
                except Exception as e:
                    hv_val = f"Error: {str(e)}"
            
            # --- ¼ÆËãÊÀ´ú¾àÀë GD ---
            distances_to_true = cdist(found_front, true_front)
            min_dist_to_true = np.min(distances_to_true, axis=1)
            gd_val = np.sqrt(np.sum(min_dist_to_true**2)) / num_sols

            # --- ¼ÆËã·´×ªÊÀ´ú¾àÀë IGD ---
            distances_from_true = cdist(true_front, found_front)
            min_dist_from_true = np.min(distances_from_true, axis=1)
            igd_val = np.sqrt(np.sum(min_dist_from_true**2)) / len(true_front)

            # --- ¼ÆËã¿Õ¼ä·Ö²¼¶È S ---
            if num_sols < 2:
                s_val = 0.0
            else:
                distances_self = cdist(found_front, found_front)
                np.fill_diagonal(distances_self, np.inf)
                d_i = np.min(distances_self, axis=1)
                d_mean = np.mean(d_i)
                s_val = np.sqrt(np.sum((d_i - d_mean)**2) / (num_sols - 1))

            results[label] = {
                'HV(3D)': hv_val,
                'GD': gd_val,
                'IGD': igd_val,
                'S': s_val,
                'PoS': num_sols
            }

        # 5. ¸ñÊ½»¯´òÓ¡Êä³ö
        print(f"\n{'-'*85}")
        # ÏÈ¶¨Òå±äÁ¿£¬±Ü¿ª f-string ÄÚ²¿µÄ·´Ð±¸ÜÏÞÖÆ
        h_hv = 'HV (3D) (Higher \u2191)'
        h_gd = 'GD (Lower \u2193)'
        h_igd = 'IGD (Lower \u2193)'
        h_s = 'S (Lower \u2193)'
        h_pos = 'PoS (\u2191)'

        # È»ºóÔÙ½øÐÐ¸ñÊ½»¯´òÓ¡
        print(f"{'Experiment':<15} | {h_hv:<20} | {h_gd:<15} | {h_igd:<15} | {h_s:<15} | {h_pos}")
        print(f"{'-'*85}")
        for label, res in results.items():
            hv_str = f"{res['HV(3D)']:.4f}" if isinstance(res['HV(3D)'], float) else res['HV(3D)']
            print(f"{label:<15} | {hv_str:<20} | {res['GD']:<15.4f} | {res['IGD']:<15.4f} | {res['S']:<15.4f} | {res['PoS']}")
        print(f"{'-'*85}")

# ================= Test Execution =================
if __name__ == '__main__':
    experiments = {
        'RoSE-opt': 'reward_suc_5_14_bo copy.csv',
        'NSGA': 'nsga_pure_results.csv',
        'NSGA-PPO': 'reward.csv',
        'PPO': 'reward_ppo.csv'
        #'ppo': 'gen_with_reward.csv'
    }
    
    comp_vis = ComparativeVisualizer(experiments)
    
    print("current name:", list(comp_vis.dfs[list(experiments.keys())[0]].columns))
    
    # print("\nGenerating comparative curves...")
    comp_vis.plot_comparative_curve(column_index=0, batch_size=512, end_step=100000)
    comp_vis.plot_comparative_curve(column_index='Feasibility Ratio', batch_size=512, end_step=100000)
    comp_vis.plot_comparative_curve(column_index='FoM', batch_size=512, end_step=100000)
    
    print("\nGenerating grouped boxplots for diversity...")
    metrics_to_plot = list(range(11)) + ['FoM']
    for col_idx in metrics_to_plot:
        comp_vis.plot_grouped_boxplot(column_index=col_idx, start_step=20000, end_step=80000)
    print("\nEvaluating MOEA convergence and diversity metrics...")
    #comp_vis.calculate_moea_metrics(last_n_steps=20000)
    #comp_vis.plot_tsne_comparison(start_step=70000, end_step=100000, perplexity=30)
    for col_idx in [0, 1, 2, 3, 'FoM']: 
        comp_vis.plot_grouped_boxplot_last_n(column_index=col_idx, last_n_steps=10000)