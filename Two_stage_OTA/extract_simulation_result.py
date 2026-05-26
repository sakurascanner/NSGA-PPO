import torch
import math
import warnings
from torch import Tensor
import numpy as np
import os
import json
from tabulate import tabulate
import gymnasium as gym
import multiprocessing as mp
import uuid
import shutil
from pathlib import Path

PSRP_target = -90
PSRN_target = -90 

TC_target = 1e-6
Power_target = 2e2
vos_target = 4e-5

cmrrdc_target = -80 
dcgain_target = 130
GBW_target = 1e6
phase_margin_target = 60 

sr_target = 4e5
settlingTime_target = 5e-6
GND = 0
Vdd = 1.8

rew_eng = True        

def ac(file_name, path):
    try:
        AMP_NMCF_ac = open(f'{path}/{file_name}', 'r')  
        lines_ac = AMP_NMCF_ac.readlines()     
        freq = []                       
        cmrrdc_ac = []
        PSRP_ac = []
        PSRN_ac = []
        dcgain_ac = []
        for line in lines_ac:
            Vac = line.split(' ')                 
            Vac = [i for i in Vac if i != '']    
            freq.append(float(Vac[0]))           
            cmrrdc_ac.append(float(Vac[1]))
            PSRP_ac.append(float(Vac[3]))
            PSRN_ac.append(float(Vac[5]))
            dcgain_ac.append(float(Vac[7]))
            
        return freq, cmrrdc_ac, PSRP_ac, PSRN_ac, dcgain_ac
    except:
        print("ac simulation errors, no .AC simulation results.")

def GBW_PM( file_name,path):
    try:
        AMP_NMCF_GBW_PM = open(f'{path}/{file_name}', 'r') 
        lines_GBW_PM = AMP_NMCF_GBW_PM.readlines()     
        freq = []                       
        GBW_ac = []
        phase_margin_ac = []
        for line in lines_GBW_PM:
            Vac = line.split(' ')                 
            Vac = [i for i in Vac if i != '']     
            freq.append(float(Vac[0]))            
            GBW_ac.append(float(Vac[1]))
            phase_margin_ac.append(float(Vac[3]))
            
        return freq, GBW_ac, phase_margin_ac
    except:
        print("gbw_pm simulation errors, no .GBW_PM simulation results.")
        
def dc( file_name,path):
    print(f'{path}/{file_name}')
    AMP_NMCF_dc = open(f'{path}/{file_name}', 'r')
    lines_dc = AMP_NMCF_dc.readlines()
    Temp_dc = []                     
    TC_dc = []
    Power_dc = []
    vos_dc = []
    for line in lines_dc:
        Vdc = line.split(' ')
        Vdc = [i for i in Vdc if i != '']
        Temp_dc.append(float(Vdc[0]))
        TC_dc.append(float(Vdc[1]))
        Power_dc.append(float(Vdc[3])) 
        vos_dc.append(float(Vdc[5]))
    
    return Temp_dc, TC_dc, Power_dc, vos_dc
    print("dc simulation errors, no .OP simulation results.")
    
def tran( file_name,path):
    try:
        AMP_NMCF_tran = open(f'{path}/{file_name}', 'r')
        lines_tran = AMP_NMCF_tran.readlines()
        time = []                         
        sr_rise = []
        sr_fall = []
        for line in lines_tran:
            line = line.split(' ')
            line = [i for i in line if i != '']
            time.append(float(line[0]))
            sr_rise.append(float(line[1]))
            sr_fall.append(float(line[3]))

        return time, sr_rise, sr_fall
    except:
            print("tran simulation errors, no .TRAN simulation results.")


def extract_tran_data(file_name,path):
    time_points = []
    raw_data = []
    vin_data = []
    vout_data = []
    time_data = []
    data_section = False
    with open(f'{path}/{file_name}', 'r')as f:
        lines = f.readlines()
        for line in lines:
            if line.strip():
                if line.startswith('Values:'):
                    data_section = True
                    continue
                if data_section:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        time_points.append(int(parts[0]))
                        raw_data.append(float(parts[1]))
                    else:
                        raw_data.append(float(parts[0]))  

    if len(time_points) != len(raw_data)/3:
        print('Error in extracting transient data')
        return None, None
    for i in time_points:
        time_data.append(raw_data[3*i])
        vin_data.append(raw_data[3*i+2])
        vout_data.append(raw_data[3*i+1])

    return time_data, vin_data, vout_data

def analyze_amplifier_performance(vinp, vout, time, d0):
    vinp = np.array(vinp)  
    vout = np.array(vout)
    time = np.array(time)
    def get_step_parameters(vinp, time):
        dv = np.diff(vinp)
        t0 = time[np.where(dv > 0)[0][0]]
        t1 = time[np.where(dv < 0)[0][0]]
        v0 = np.median(vinp[time < t0])
        v1 = np.median(vinp[(time > t0) & (time < t1)])
        return v0, v1, t0, t1
    v0, v1, t0, t1 = get_step_parameters(vinp, time)

    pre_step_data = vout[time < t0]
    delta0 = (pre_step_data - v0) / v0
    d0_settle = np.mean(np.abs(delta0))
    stable = not np.any(np.abs(delta0) > d0)

    def find_settling_time_index(delta, d0):
        for i in range(len(delta)):
            if np.all(np.abs(delta[i:]) < d0):
                return i
        return None

    def get_slope_and_settling_time(vout, time, v0, v1, start_t, end_t, d0, mode):
        idx = (time >= start_t) & (time <= end_t)
        vout_segment = vout[idx]
        time_segment = time[idx]

        target_value = v0 + (v1 - v0) / 2
        idx_target = np.where(vout_segment >= target_value)[0][0] if np.any(vout_segment >= target_value) else None
        if idx_target is None:
            SR = np.nan
        else:
            SR = np.gradient(vout_segment, time_segment)[idx_target]

        if mode == 'positive':
            delta = (vout_segment - v1) / v1
        else:
            delta = (vout_segment - v0) / v0

        idx_settle = find_settling_time_index(delta, d0)
        if idx_settle is None:
            settling_time = np.nan
            d_settle = np.mean(np.abs(delta))
        else:
            settling_time = time_segment[idx_settle] - start_t
            d_settle = np.mean(np.abs(delta[idx_settle:]))
        return SR, settling_time, d_settle

    SR_p, settling_time_p, d1_settle = get_slope_and_settling_time(vout, time, v0, v1, t0, t1, d0, 'positive')

    SR_n, settling_time_n, d2_settle = get_slope_and_settling_time(vout, time, v0, v1, t1, np.max(time), d0, 'negative')

    return d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n 

def _get_info(path):
    '''Evaluate the performance'''
    ''' DC '''
    dc_results = dc(file_name='AMP_NMCF_ACDC_DC',path=path)
    TC = dc_results[1][1]
    Power = dc_results[2][1]
    vos_1 = dc_results[3][1]
    vos = abs(vos_1)
            
    TC_score = np.max(np.min([(TC_target - TC) / (TC_target + TC), 0]),-1)
    Power_score = np.max(np.min([(Power_target - Power) / (Power_target + Power), 0]),-1)
    vos_score = np.max(np.min([(vos_target - vos) / (vos_target + vos), 0]),-1)

    ''' AC '''
    ac_results = ac(file_name='AMP_NMCF_ACDC_AC',path=path)
    cmrrdc = ac_results[1][1]
    if cmrrdc > 0 :
        cmrrdc_score = -1
    else : 
        cmrrdc_score = np.max(np.min([(cmrrdc - cmrrdc_target) / (cmrrdc + cmrrdc_target), 0]),-1)
        if cmrrdc < cmrrdc_target:
            cmrrdc_score = 0

    PSRP = ac_results[2][1]
    if PSRP > 0 :
        PSRP_score = -1
    else : 
        PSRP_score = np.max(np.min([(PSRP - PSRP_target) / (PSRP + PSRP_target), 0]),-1)
        if PSRP < PSRP_target:
            PSRP_score = 0

    PSRN = ac_results[3][1]
    if PSRN > 0 :
        PSRN_score = -1
    else : 
        PSRN_score = np.max(np.min([(PSRN - PSRN_target) / (PSRN + PSRN_target), 0]),-1)
        if PSRN < PSRN_target:
            PSRN_score = 0

    dcgain = ac_results[4][1]
    if dcgain > 0 :
        try:
            dcgain_score = np.clip((dcgain - dcgain_target) / (dcgain + dcgain_target),-1,0)
            GBW_PM_results = GBW_PM(file_name='AMP_NMCF_ACDC_GBW_PM',path=path)
            GBW = GBW_PM_results[1][1]
            GBW_score = np.clip((GBW - GBW_target) / (GBW + GBW_target),-1,0)
            phase_margin = GBW_PM_results[2][1]
            phase_margin_score = np.clip((phase_margin - phase_margin_target) / (phase_margin + phase_margin_target),-1,0)
        except: 
            if phase_margin > 180 or phase_margin < 0:
                phase_margin = 0
    else :
        dcgain_score = -1
        GBW = 0
        GBW_score = -1
        phase_margin = 0
        phase_margin_score = -1
    
    """ Tran """
    tran_results = tran(file_name='AMP_NMCF_Tran',path=path)
    sr_rise = tran_results[1][1]
    sr_fall = tran_results[2][1]
    sr = (sr_rise + sr_fall) / 2 
    sr_score = np.max(np.min([(sr - sr_target) / (sr + sr_target), 0]),-1)

    """ setting_time """
    meas = {}
    d0 = 0.01
    # path = './benchmarks/TB_Amplifier_ACDC/'
    time_data, vin_data, vout_data = extract_tran_data(file_name='tran.dat',path=path)
    if time_data is None:
        return None
    d0_settle, d1_settle, d2_settle, stable, SR_p, settling_time_p, SR_n, settling_time_n = analyze_amplifier_performance(vin_data, vout_data, time_data, d0)
    d0_settle = abs(d0_settle)
    d1_settle = abs(d1_settle)
    d2_settle = abs(d2_settle)
    SR_n = abs(SR_n)
    SR_p = abs(SR_p)
    settlingTime_p = abs(settling_time_p)
    settlingTime_n = abs(settling_time_n)

    if math.isnan(d0_settle):
        d0_settle = 10

    if math.isnan(d1_settle) or math.isnan(d2_settle) :
        if math.isnan(d1_settle):
            d0_settle += 10
        if math.isnan(d2_settle):
            d0_settle += 10
        d_settle = d0_settle
    else:
        d_settle = max(d0_settle, d1_settle, d2_settle)

    if math.isnan(SR_p) or math.isnan(SR_n) :
        SR = -d_settle
    else:
        SR = min(SR_p,SR_n)

    if math.isnan(settlingTime_p) or math.isnan(settlingTime_n) :
        settlingTime = d_settle
    else:
        settlingTime = max(settlingTime_p, settlingTime_n)
    
    meas['d_settle'] = d_settle
    meas['SR'] = SR
    meas['settlingTime'] = settlingTime
    settlingTime_score = np.max(np.min([(settlingTime_target - settlingTime) / (settlingTime_target + settlingTime), 0]),-1)

    if sr_score < -1:
        sr_score = -1

    """ Total reward """
    reward = TC_score + Power_score + vos_score + cmrrdc_score + \
                    dcgain_score + GBW_score + phase_margin_score + PSRP_score + \
                    PSRN_score + sr_score + settlingTime_score 
    
    return {
        'TC': [TC, TC_score], 
        'Power': [Power, Power_score], 
        'vos': [vos, vos_score], 
        'cmrrdc': [cmrrdc, cmrrdc_score], 
        'dcgain': [dcgain, dcgain_score], 

        'GBW': [GBW, GBW_score], 
        'phase_margin (deg)': [phase_margin, phase_margin_score], 
        'PSRP': [PSRP, PSRP_score], 
        'PSRN': [PSRN, PSRN_score], 

        'sr': [sr, sr_score], 
        'settlingTime': [settlingTime, settlingTime_score],
        'reward': reward
    }


simulations_dir = Path("/home/tanjunfeng/New_RL_expr/simulations")

successful_folders = []

for folder in simulations_dir.iterdir():
    if folder.is_dir(): 
        tran_dat = folder / "tran.dat"
        amp_file = folder / "AMP_NMCF_ACDC_GBW_PM"
        if  tran_dat.exists() and amp_file.exists():
            successful_folders.append(folder.name)

print("Simulation success folder:")
for name in successful_folders:
    print(name)
print(_get_info('/home/tanjunfeng/New_RL_expr/simulations/0a1ca1e3'))