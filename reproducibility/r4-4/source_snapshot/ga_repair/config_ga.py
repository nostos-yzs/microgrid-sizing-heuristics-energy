# -*- coding: utf-8 -*-
"""
config_ga.py  —  最终对齐版
- 三个实验（Simple-GA / Guided-GA / PSO）除“能流诊断注入方式”外，其余参数完全一致。
- SAC 仅调整引导（guide）相关参数：treat_prob（父代治疗比例）、gamma_feedback（父代校正强度）。
"""

import numpy as np
import torch

def get_config():
    """
    返回包含所有 GA/PSO 对比实验公共参数的配置字典。
    """
    config = {
        # ------------------ 微电网环境参数（与 PSO 保持一致） ------------------
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

        # 典型日分布（如需）
        "typical_day_distribution": [50, 60, 70, 85, 100],

        # ------------------ GA & 模拟参数（对齐三实验） ------------------
        "ga_params": {
            "population_size": 150,
            "min_bounds": np.array([0, 0, 0, 0, 0, 0]),
            "max_bounds": np.array([1000, 1000, 200, 200, 1000, 1000]),

            "classic_params": {
                # Simple-GA：不使用诊断，骨架参数与 Guided-GA 完全一致
                "simple": {
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.4,
                    "mutation_strength": 0.1
                },
                # Guided-GA：父代先校正（PSO 同式）+ 经典 GA
                # 注意：与 simple 的 GA 参数完全一致，仅多了父代治疗参数（默认值）
                "guided": {
                    "crossover_rate": 0.9,              # 与 simple 同值
                    "mutation_rate": 0.4,               # 与 simple 同值
                    "random_mutation_strength": 0.1,    # = simple.mutation_strength
                    "parent_treat_prob": 0.1,           # 父代治疗比例（SAC不启用时默认值）
                    "parent_treat_strength": 1.5        # 父代校正强度（= PSO 的 gamma_feedback）
                }
            }
        },

        "simulation_params": {
            "total_iterations": 250,   # GA 代数 / PSO 迭代数
            "num_independent_runs": 1
        },

        # 随机扰动（若需要蒙特卡洛）
        "stochasticity_params": {
            "enabled": False, "mc_samples": 1,
            "load_std_dev_percent": 0.05,
            "ghi_std_dev_percent": 0.10,
            "wind_std_dev_percent": 0.15,
        },

        # ------------------ SAC（仅调 guide 参数） ------------------
        "sac_agent_params": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "state_dim": 4,

            # 动作：仅两维 [treat_prob, gamma_feedback]
            "action_dim": 2,
            "action_bounds": {
                "treat_prob":     [0.01, 0.21],   # 父代被治疗的比例
                "gamma_feedback": [1, 2]   # 父代校正强度（等价 PSO 的 γ_feedback）
            },

            "buffer_size": 20000,
            "batch_size": 128,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "actor_lr": 3e-5,
            "critic_lr": 3e-5,

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
    }
    return config
