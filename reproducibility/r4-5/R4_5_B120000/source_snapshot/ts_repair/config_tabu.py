# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:54:34 2025

@author: Lenovo
"""

# ts/config_tabu.py
import numpy as np
import torch

def get_config():
    config = {
        # —— 微电网参数：与你 GA 的一致（可直接复制 GA 里的 microgrid_params）——
            # ------------------ 微电网环境参数 (与之前版本完全相同) ------------------
        "microgrid_params": {
            #microgrid tech
            "discount_rate": 0.05, "lifetime": 20, 
            
            #tech
            "G_ref": 1000, "T_ref": 25, "NOCT": 45, "alpha_P": -0.0047,
            "eta_ch": 0.95, "eta_dis": 0.95,"C_BESS_initial": 10,  
            "eta_EC_initial": 0.6, "eta_FC_initial": 0.5,"HHV": 39.4,
            "single_pv_rated_power": 3, "f_pv": 0.9,
            "wind_hub_height": 50, "wind_ref_height": 10, "wind_alpha": 0.14, "wind_rated_speed": 12, "wind_cut_in_speed": 3, "wind_cut_out_speed": 25, 
            
            #rated
            "P_ch_rated": 5,"P_dis_rated": 5, 
            "single_tank_capacity_kg": 100,     
            "single_wind_rated_power": 10, 
            "single_ec_rated_power": 5,"single_fc_rated_power": 5, 
            "SOH_min": 0.1, "SOH_max": 0.9,
            "SOC_min": 0.2,"SOC_max": 0.8,
            #economic
            "C_RES_cap": 800,"C_RES_OM": 20, 
            "C_WIND_cap": 1500, "C_WIND_OM": 30,
            "C_BESS_cap": 400,"C_BESS_OM": 0,"C_BESS_OM_var":0.03,
            "C_FC_cap": 800,"C_FC_OM": 0,"C_FC_OM_var": 0.02,      
            "C_EC_cap": 600,"C_EC_OM": 0,"C_EC_OM_var": 0.015,  
            "C_HSS_OM_var": 0.02,"C_TANK_cap": 12000,

            #买电买氢没用了
            "buy_back_prices": {"electricity": 0.165, "hydrogen": 4.6},
            #退化参数
            "BESS_EoL_SOH": 0.8, "BESS_fade_per_throughput": 1.5e-5, "SO_EoL_eff_factor": 0.85,
            "SO_degradation": {
                "d0_fc": 0.00002, "d0_ec": 0.00002, "gamma_fc": 1.5, "gamma_ec": 1.5,
                "Ea": 1.1e5, "R": 8.314, "T_ref": 800 + 273.15,
                "T_fc_op": 750 + 273.15, "T_ec_op": 750 + 273.15
            },
            # ============================
            # 新退化模型参数（基于你论文表格）
            # ============================

            # FC degradation model
            "fc_deg": {
                "s_chg": 0.04185e-6,
                "s_low": 8.622e-6,
                "s_high": 10e-6,
                "s_ss": 13.79e-6
            },

            # EC degradation model
            "ec_deg": {
                "s_op": 32e-6,
                "s_ss": 30e-6
            },

            # Battery degradation model
            "bat_deg": {
                "k1": 4.83e-4,
                "k2": 2.38e-5,
                "k3": 4.34e-6,
                "k4": 2.73e-5,
                "k5": 1.45e-5
            },
        },

        "typical_day_distribution": [50, 60, 70, 85, 100],

        # —— TS 参数（含边界）——
        "ts_params": {
            "min_bounds": np.array([0, 0, 0, 0, 0, 0]),
            "max_bounds": np.array([1000, 1000, 200, 200, 1000, 1000]),
            "neighbors": 16,          # 每轮邻域候选数 K
            "tabu_len": 32,           # 禁忌表长度
            "sigma_random": 0.15,     # 随机邻域步幅（占 span 比例）
            "grid_ratio": 0.01,       # 区域哈希网格尺度（占 span）
            "tau_ratio": 0.005,       # 显著移动阈值（占 span）
            "guided": {
                "diag_prob": 0.75,     # 经典引导比例（非 SAC）
                "gamma_feedback": 70.5
            }
        },

        "simulation_params": {
            "total_iterations": 200,
            "num_independent_runs": 1
        },
        
        "random_seed": 42,

        # —— SAC：仅两维动作（与 GA 的不同，名称对应 TS 含义）——
        "sac_agent_params": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "state_dim": 4,
            "action_dim": 2,
            "action_bounds": {
                "diag_prob":     [0.65, 0.85],   # 诊断邻域比例
                "gamma_feedback":[68.5,72.5]  # 诊断强度缩放
            },
            "buffer_size": 20000,
            "batch_size": 128,
            "gamma": 0.99, "tau": 0.005, "alpha": 0.2,
            "actor_lr": 3e-5, "critic_lr": 3e-5,

            "rewards": {
                "gbest_improvement": 10,
                "stagnation": -10,
                "first_feasible_bonus": 50,
            },
            "penalties": {
                "fail_supply": -20,
                "fail_storage_overdischarge": -20,
                "fail_storage_overcharge": -20,
            }
        },

        # 若要启用随机性，与 GA 相同配置
        "stochasticity_params": {
            "enabled": False, "mc_samples": 1,
            "load_std_dev_percent": 0.05,
            "ghi_std_dev_percent": 0.10,
            "wind_std_dev_percent": 0.15,
        },
    }
    return config
