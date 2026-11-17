# -*- coding: utf-8 -*-
"""
config_sa.py —— SA 三模式统一配置（严格沿用 microgrid/典型日/随机性参数）
仅在 SA 自身超参上做最小配置；使用“外层降温 × 内层采样”的两层结构。
"""

import numpy as np
import torch


def get_config():
    # —— 与对比实验一致的基础设置（不影响微电网口径）——
    RANDOM_SEED = 10

    # 决策变量顺序：PV, WIND, EC, FC, TANK, BESS（仅作初始化边界）
    MIN_BOUNDS = np.array([0, 0,  0,  0,  0, 0], dtype=float)
    MAX_BOUNDS = np.array([1000, 1000, 200, 200, 1000, 1000], dtype=float)


    # === 微电网参数（与原始版本一致，控制变量用，严禁改动口径）===
    microgrid_params = {
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
    }

    # —— 典型日分布（这里是“天数”，5类日合计=365）——
    typical_day_distribution = [50, 60, 70, 85, 100]

    # —— 随机性参数（保持一致）——
    stochasticity_params = {
        "enabled": False, "mc_samples": 8,
        "load_std_dev_percent": 0.05,
        "ghi_std_dev_percent": 0.10,
        "wind_std_dev_percent": 0.15,
    }

    # —— SA 三模式（仅影响 SA 搜索逻辑，不动微电网参数）——

    sac_agent_params = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "state_dim": 4,
        "action_dim": 2,  # (gamma_fb, diag_prob)
        "action_bounds": {
            "gamma_fb":  [1.8, 3.8],#[1.8, 3.8],
            "diag_prob": [0.26, 0.46],#[0.26, 0.46],
        },
        "rewards": {"gbest_improvement": 10, "stagnation": -10, "first_feasible_bonus": 50},
        "penalties": {"fail_supply": -20, "fail_storage_overdischarge": -20, "fail_storage_overcharge": -20},
        "buffer_size": 20000, "batch_size": 128,
        "gamma": 0.99, "tau": 0.005, "alpha": 0.2,
        "actor_lr": 3e-5, "critic_lr": 3e-5,
    }

    # —— 两层参数（总步数 = n_layers * inner_loops）——
    # 为了与原来总步数 30 对齐，这里示例给出 n_layers=30, inner_loops=1
    N_LAYERS = 60
    INNER_LOOPS = 60
    TOTAL_ITERS = N_LAYERS * INNER_LOOPS

    SA_SCHEDULE = {"T0": 100.0, "alpha": 0.998, "Tmin": 1e-3}
    
    experiments = {
        "classic_simple": {
            "mode": "classic_simple",
            "sa_schedule": SA_SCHEDULE,
            "random_seed": RANDOM_SEED,
            "total_iters": TOTAL_ITERS,
        },
        "classic_guided": {
            "mode": "classic_guided",
            "gamma_fb": 2.8,#2.8,
            "diag_prob": 0.36,#0.36,
            "sa_schedule": SA_SCHEDULE,
            "random_seed": RANDOM_SEED,
            "total_iters": TOTAL_ITERS,
        },
        "sac_guided": {
            "mode": "sac_guided",
            "sa_schedule": SA_SCHEDULE,
            "random_seed": RANDOM_SEED,
            "sac_agent_params": sac_agent_params,
            "total_iters": TOTAL_ITERS,

        },
    }

    # —— 统一返回（微电网口径与你提供的一致）——
    return {
        "random_seed": RANDOM_SEED,
        "min_bounds": MIN_BOUNDS,
        "max_bounds": MAX_BOUNDS,

        "microgrid_params": microgrid_params,
        "typical_day_distribution": typical_day_distribution,
        "stochasticity_params": stochasticity_params,

        "sa_params": {
            "schedule": SA_SCHEDULE,
            "n_layers": N_LAYERS,
            "inner_loops": INNER_LOOPS,
            "total_iters": TOTAL_ITERS,  # 仅用于统计/日志
        },
        "experiments": experiments,

        "simulation_params": {
            "base_seed": 10,
            "total_iterations": TOTAL_ITERS,  # RL 状态比例等使用
            "n_layers": N_LAYERS,
            "inner_loops": INNER_LOOPS,
            "num_independent_runs": 1,
            "random_seed": RANDOM_SEED,
        },
    }
