import torch
import numpy as np
import os
"""
Here you define the graph for a circuit
"""

class GraphAMPNMCF:
    """                                                                                                                           

    node 0 : M0 , node 1 : M1 , node 2 : M2 , node 3 : M3 , node 4 : M4 , node 5 : M5
    node 6 : M6 , node 7 : M7 , node 8 : M8 , node 9 : M9 , node 10 : M10 , node 11 : M11
    node 12 : M12 , node 13 : M13 , node 14 : M14 , node 15 : M15 , node 16 : M16 , node17 : M17 ,
    node 18 : M18 , node 19 : M19 , node 20 : M20 , node 21 : M21 , node 22 : M22 ,   
    node23 : M23 , node24 : Ib , node25 : VDD , node26 : GND , node27 : C0 , node28 : C1

    """
    def __init__(self):        
        self.device = torch.device("cpu") # »ò "cuda"

        self.action_space_low = np.array([
            2.0,    # W_diff: ÊäÈë¶Ô²»ÄÜÌ«Ð¡£¬·ñÔòÔëÉùºÍÊ§Åä´ó
            0.5,    # L_diff
            1.0,    # W_load1
            0.5,    # L_load1
            1.0,    # W_pmos_mirror
            0.5,    # L_pmos_mirror
            5.0,    # W_gm2: ±ØÐë×ã¹»´óÒÔÌá¹© Gm
            0.15,   # L_gm2: ÔÊÐíÊ¹ÓÃ¹¤ÒÕ×îÐ¡Öµ (150nm) »»È¡ËÙ¶È
            2,      # M_out_stage: ÖÁÉÙÊÇ»ù×¼µçÁ÷µÄ2±¶
            0.5,    # W_bias_n
            1.0,    # L_bias_n: Æ«ÖÃµçÂ·½¨Òé³¤Ò»µã£¬Ôö¼ÓÊä³ö×è¿¹
            0.5,    # W_Rc
            0.1e-12,# Cc_val: ×îÐ¡ 0.1pF
            1.0e-6  # current_0_bias: ×îÐ¡ 1uA
        ])

        # 1. ¶¨ÒåµçÂ·²ã¼¶ÓëÆ÷¼þÓ³Éä
        # Format: ('Symbol', 'SPICE_Name', 'Model', 'Type')

        # ½Úµã¶¨Òå (¹²22¸ö½Úµã: 20¸öÆ÷¼þ + VDD + GND)
        self.ckt_hierarchy = (
            ('M1', 'x1.XM1', 'pfet_01v8', 'm'),        # 0
            ('M2', 'x1.XM2', 'pfet_01v8', 'm'),        # 1
            ('M3', 'x1.XM3', 'nfet_01v8', 'm'),        # 2
            ('M4', 'x1.XM4', 'nfet_01v8', 'm'),        # 3
            ('M5', 'x1.XM5', 'pfet_01v8', 'm'),        # 4
            ('M6', 'x1.XM6', 'nfet_01v8', 'm'),        # 5
            ('M7', 'x1.XM7', 'pfet_01v8', 'm'),        # 6
            
            ('M_master', 'x1.XM_master', 'pfet_01v8', 'm'), # 7
            ('M_feeder', 'x1.XM_feeder', 'pfet_01v8', 'm'), # 8
            ('M14', 'x1.XM14', 'nfet_01v8', 'm'),      # 9
            ('M16', 'x1.XM16', 'nfet_01v8', 'm'),      # 10
            ('M8',  'x1.XM8',  'pfet_01v8', 'm'),      # 11
            ('M13', 'x1.XM13', 'nfet_01v8', 'm'),      # 12
            ('M15', 'x1.XM15', 'nfet_01v8', 'm'),      # 13
            
            ('M10', 'x1.XM10', 'pfet_01v8', 'm'),      # 14
            ('M11', 'x1.XM11', 'nfet_01v8', 'm'),      # 15
            ('M12', 'x1.XM12', 'nfet_01v8', 'm'),      # 16
            ('M9',  'x1.XM9',  'nfet_01v8', 'm'),      # 17 (Rc)

            ('Ib', '', 'Ib', 'i'),                     # 18
            ('Cc', 'x1.XCc', 'cap_mim_m3_1', 'c'),      # 19
        )
        
        # ÐéÄâµçÔ´½Úµã (²»°üº¬ÔÚ ckt_hierarchy ÖÐ£¬µ«ÓÃÓÚ edge_index)
        # Node 20: VDD, Node 21: GND

        self.op = {name: {} for name, _, _, _ in self.ckt_hierarchy}

        # --- Á¬½Ó¹ØÏµ¶¨Òå (Adjacency Matrix) ---
        edges = [
            # 1. ÐÅºÅÂ·¾¶ (Signal Path)
            # Diff Pair Input/Tail
            [4, 0], [0, 4], [4, 1], [1, 4], [0, 1], [1, 0],
            # Active Load (Current Mirror)
            [0, 2], [2, 0], [2, 3], [3, 2], [1, 3], [3, 1],
            # Stage 1 to Stage 2
            [3, 5], [5, 3], [3, 17], [17, 3], # M4 Drain -> M6 Gate & M9(Rc) Source
            # Output Node
            [5, 6], [6, 5], [5, 19], [19, 5], [6, 19], [19, 6],
            # Compensation
            [17, 19], [19, 17], # M9 Drain -> Cc

            # 2. Æ«ÖÃÓë¾µÏñ¹ØÏµ (Bias & Mirrors)
            # Ib Loop: Ib(18) Á¬½ÓËùÓÐ PMOS Mirror µÄ Gate
            [18, 7], [7, 18], [18, 4], [4, 18], [18, 6], [6, 18], 
            [18, 8], [8, 18], [18, 14], [14, 18],

            # NMOS Bias Generation (Bottom Left)
            # M_feeder(8) -> M14(9)
            [8, 9], [9, 8],
            # Stack: M14(9) -> M16(10)
            [9, 10], [10, 9],
            
            # *** M13/M15 µÄÁ¬½Ó (Äã¹Ø×¢µÄÖØµã) ***
            # Mirror Gates: M14(9) Gate -> M13(12) Gate
            [9, 12], [12, 9],
            # Mirror Gates: M16(10) Gate -> M15(13) Gate
            [10, 13], [13, 10],
            # Stack: M13(12) Source -> M15(13) Drain
            [12, 13], [13, 12],
            # M8(11) Drain -> M13(12) Drain
            [11, 12], [12, 11],

            # Rc Control Logic
            [14, 15], [15, 14], # M10 -> M11
            [15, 16], [16, 15], # M11 -> M12
            [15, 17], [17, 15], # M11 Gate -> M9 Gate (Control Voltage)

            # 3. µçÔ´Á¬½Ó (VDD=20, GND=21)
            # PMOS Sources -> VDD
            [7, 20], [4, 20], [6, 20], [8, 20], [11, 20], [14, 20], [0, 20], [1, 20],
            # NMOS Sources -> GND
            [2, 21], [3, 21], [5, 21], [10, 21], [13, 21], [16, 21]
        ]

        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        self.num_nodes = 22

        # 3. ¶¨Òå Edge Type (0: Signal/Data path, 1: Bias/Power path)
        # ÎªÁË¼ò»¯£¬ÕâÀïÍ³Ò»³õÊ¼»¯Îª 0£¬Èç¹ûÄãµÄ GNN ¶Ô±ßÀàÐÍÃô¸Ð£¬ÐèÒª°´Ë÷ÒýÏ¸·Ö
        self.edge_type = torch.zeros(self.edge_index.size(1), dtype=torch.long).to(self.device)

        self.num_relations = 2
        self.num_nodes = 22 # 0~19 Devices + VDD + GND
        self.num_node_features = 12 # ±£³ÖÄãÔ­ÓÐµÄÌØÕ÷Î¬¶È
        self.obs_shape = (self.num_nodes, self.num_node_features)

        """Select an action from the input state."""

        self.W_C0 = 30
        self.L_C0 = 30
        M_C0_low = 1
        M_C0_high = 50
        self.C0_low = M_C0_low * (self.L_C0 * self.W_C0 * 2e-15 + (self.L_C0 + self.W_C0) *0.38e-15)
        self.C0_high = M_C0_high * (self.L_C0 * self.W_C0 * 2e-15 + (self.L_C0 + self.W_C0)*0.38e-15)
        
        
        self.action_dim = len(self.action_space_low)
        self.action_shape = (self.action_dim,)    
        
        """Some target specifications for the final design"""
        self.PSRP_target = -70
        self.PSRN_target = -70 
        
        self.TC_target = 10e-6
        self.Power_target = 10e2
        self.vos_target = 10e-5
        
        self.cmrrdc_target = -80 
        self.dcgain_target = 90
        self.GBW_target = 1e6
        self.phase_margin_target = 60 

        self.sr_target = 4e5
        self.settlingTime_target = 5e-6
        self.GND = 0
        self.Vdd = 1.8
        
        self.rew_eng = True        