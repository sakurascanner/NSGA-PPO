import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
import numpy as np
import os
from scipy.spatial.distance import cdist

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
            'Power', 'DC Gain', 'GBW', 'Phase Margin', 
            'TC', 'Vos', 'CMRR', 'PSRP', 
            'PSRN', 'SR', 'Settling Time', 'Reward'
        ]
        
        self.targets = {
            'Power': 10e2,           
            'DC Gain': 90,           
            'GBW': 1e6,              
            'Phase Margin': 60,      
            'TC': 10e-6,             
            'Vos': 10e-5,            
            'CMRR': -80,             
            'PSRP': -70,             
            'PSRN': -70,             
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

    def _calculate_feasibility(self, df):
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
        return ratio

    def _calculate_fom(self, df):
        fom = pd.Series(0.0, index=df.index)
        eps = 1e-12

        maximize_cols = ['DC Gain', 'GBW', 'Phase Margin', 'SR', 'CMRR', 'PSRP', 'PSRN']
        for col in maximize_cols:
            T = self.targets[col]
            V = df[col]
            fom += (V - T) / (V + T + eps)

        minimize_cols = ['Power', 'TC', 'Settling Time']
        for col in minimize_cols:
            T = self.targets[col]
            V = df[col]
            fom += (T - V) / (T + V + eps)
            
        T_vos = self.targets['Vos']
        V_vos = df['Vos'].abs() 
        fom += (T_vos - V_vos) / (T_vos + V_vos + eps)

        failed_mask = df['Power'].isna()
        fom[failed_mask] = np.nan
        
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
            
            df['Feasibility Ratio'] = self._calculate_feasibility(df)
            df['FoM'] = self._calculate_fom(df) 
            
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
            
            x_data = np.array(batch_mean.index + 1, dtype=np.float64).flatten()
            y_mean = np.array(batch_mean[metric], dtype=np.float64).flatten()
            y_std = np.array(batch_std[metric], dtype=np.float64).flatten()
            
            y_mean = np.nan_to_num(y_mean, nan=0.0)
            y_std = np.nan_to_num(y_std, nan=0.0)

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

    def _is_pareto_efficient(self, costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  
                is_efficient[i] = True
        return is_efficient

    def calculate_moea_metrics(self, last_n_steps=10000):
        # Ô­Âß¼­±£³Ö²»±ä
        pass 

# ================= Test Execution =================
if __name__ == '__main__':
    experiments = {
        '12-512': 'reward_suc_5_8 copy.csv',
        '96-64': '_suc_5_14_96ref_64step.csv',
        '384-16': 'reward.csv',
    }
    
    comp_vis = ComparativeVisualizer(experiments)
    
    print("current name:", list(comp_vis.dfs[list(experiments.keys())[0]].columns))
    
    print("\nGenerating comparative curves...")
    comp_vis.plot_comparative_curve(column_index=11, batch_size=512, end_step=200000)
    comp_vis.plot_comparative_curve(column_index='Feasibility Ratio', batch_size=512, end_step=200000)
    comp_vis.plot_comparative_curve(column_index='FoM', batch_size=512, end_step=200000)
    
    print("\nGenerating grouped boxplots for diversity...")
    metrics_to_plot = list(range(11)) + ['FoM']
    for col_idx in metrics_to_plot:
        comp_vis.plot_grouped_boxplot(column_index=col_idx, start_step=60000, end_step=120000)

    # for col_idx in [0, 1, 2, 3, 'FoM']: 
    #     comp_vis.plot_grouped_boxplot_last_n(column_index=col_idx, last_n_steps=10000)