# microgrid_env.py
import numpy as np
import pandas as pd

class MicrogridEnv:
    """
    微电网环境类，采用“年度退化外推”策略，并支持蒙特卡洛模拟以处理不确定性。
    """
    def __init__(self, config, typical_days_data):
        self.config = config
        self.params = config["microgrid_params"]
        self.stochastic_params = config.get("stochasticity_params", {"enabled": False})
        self.typical_days_data = typical_days_data
        self.day_distribution = config["typical_day_distribution"]

    def evaluate_solution(self, solution):
        """
        主评估函数。如果启用了随机性，则运行蒙特卡洛模拟并返回平均结果。
        """
        if not self.stochastic_params.get("enabled", False):
            # 如果未启用随机性，则直接运行一次确定性模拟
            return self._run_single_simulation(solution, stochastic=False)

        lcoe_samples = []
        num_samples = self.stochastic_params.get("mc_samples", 1)

        for _ in range(num_samples):
            lcoe, diagnostics = self._run_single_simulation(solution, stochastic=True)
            lcoe_samples.append(lcoe)
        
        final_lcoe = np.mean(lcoe_samples)
        final_diagnostics = {"status": "success_mc_avg"} if final_lcoe < 1e6 else {"status": "fail_mc_avg"}
        
        return final_lcoe, final_diagnostics

    def _run_single_simulation(self, solution, stochastic=False):
        """
        运行一次完整的“平均年”模拟。
        终点规则：多于额定上限=overcharge；少于期初=overdischarge；不计成本，直接不可行。
        """
        try:
            p = self.params
            n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in solution)
            # if n_PV < 1 and n_WIND < 1: n_PV = 1
            # if n_EC < 1: n_EC = 1
            # if n_FC < 1: n_FC = 1
            # if n_TANK < 1: n_TANK = 1
    
            annual_buy_back_cost = 0.0
            annual_hss_om_cost   = 0.0
            annual_bess_om_cost  = 0.0
    
            diagnostics = {"status": "success", "delta_E": 0.0, "delta_P": 0.0}
    
            # 必须包含 calendar 和 cycle 两类退化
            annual_metrics = {
                "bess_throughput": 0.0,
                "fc_fade": 0.0,
                "ec_fade": 0.0,
                "bess_cal_deg": 0.0,
                "bess_cyc_deg": 0.0
            }
    
            # 遍历典型日
            for i, day_df_original in enumerate(self.typical_days_data):
                day_df = day_df_original.copy()
                days_this_type = self.day_distribution[i]
    
                # 随机扰动
                if stochastic and self.stochastic_params.get("enabled", False):
                    s_params = self.stochastic_params
                    day_df['GHI']   = np.maximum(0, np.random.normal(day_df['GHI'],   day_df['GHI']   * s_params['ghi_std_dev_percent']))
                    day_df['Vwind'] = np.maximum(0, np.random.normal(day_df['Vwind'], day_df['Vwind'] * s_params['wind_std_dev_percent']))
                    day_df['Load']  = np.maximum(0, np.random.normal(day_df['Load'],  day_df['Load']  * s_params['load_std_dev_percent']))
    
                # 设备边界
                total_tank_capacity_kg = n_TANK * p['single_tank_capacity_kg']
                min_h_storage_kg = total_tank_capacity_kg * p['SOH_min']
                max_h_storage_kg = total_tank_capacity_kg * p['SOH_max']
    
                P_PV_t   = self._calculate_pv_power(day_df, n_PV)
                P_WIND_t = self._calculate_wind_power(day_df, n_WIND)
                P_RES_t  = P_PV_t + P_WIND_t
                P_load_t = day_df['Load'].values
    
                # 期初状态
                initial_h_storage_kg = 0.5 * (min_h_storage_kg + max_h_storage_kg)
                initial_soc          = 0.5 * (p['SOC_min'] + p['SOC_max'])
    
                current_h_storage_kg = initial_h_storage_kg
                soc = initial_soc
    
                # === initialize degradation states ===
                p_fc_prev = 0.0
                p_ec_prev = 0.0
                prev_dir = 0
                soc_cycle_list = []
    
                # 逐时仿真
                for t in range(24):
                    net_power = float(P_RES_t[t] - P_load_t[t])
                    p_ec = p_fc = p_ch = p_dis = 0.0
    
                    if net_power > 0:
                        # 富余功率
                        p_ec = min(net_power, n_EC * p['single_ec_rated_power'])
                        p_ch = min(net_power - p_ec, n_BESS * p['P_ch_rated']) if n_BESS > 0 else 0.0
                    else:
                        # 功率缺口
                        demand = -net_power
                        p_fc  = min(demand, n_FC * p['single_fc_rated_power'])
                        p_dis = min(demand - p_fc, n_BESS * p['P_dis_rated']) if n_BESS > 0 else 0.0
                        power_deficit = demand - p_fc - p_dis
                        if power_deficit > 1e-2:
                            diagnostics = {"status": "fail_supply", "delta_P": float(power_deficit)}
                            return 1e6 + abs(1000.0 * power_deficit), diagnostics
    
                    # 状态更新 (H2)
                    if p['HHV'] > 0:
                        h_produced_kg = (p_ec * p['eta_EC_initial']) / p['HHV']
                        h_consumed_kg = (p_fc / p['eta_FC_initial']) / p['HHV'] if p['eta_FC_initial'] > 0 else float('inf')
                    else:
                        h_produced_kg = 0.0
                        h_consumed_kg = 0.0
    
                    current_h_storage_kg += (h_produced_kg - h_consumed_kg)
    
                    # 状态更新 (SOC)
                    soc_prev = soc  # calendar 退化必须记录
                    if n_BESS > 0 and p.get("C_BESS_initial", 0) > 0:
                        soc += (p_ch * p['eta_ch'] - p_dis / p['eta_dis']) / (n_BESS * p["C_BESS_initial"])
    
                    # 越界检查
                    if current_h_storage_kg > max_h_storage_kg:
                        return 1e6 + abs(500.0 * (current_h_storage_kg - max_h_storage_kg)), {
                            "status": "fail_storage_overcharge", "which": "H2",
                            "delta": float(current_h_storage_kg - max_h_storage_kg)
                        }
                    if current_h_storage_kg < min_h_storage_kg:
                        return 1e6 + abs(500.0 * (min_h_storage_kg - current_h_storage_kg)), {
                            "status": "fail_storage_overdischarge", "which": "H2",
                            "delta": float(min_h_storage_kg - current_h_storage_kg)
                        }
    
                    if n_BESS > 0:
                        if soc > p['SOC_max']:
                            return 1e6 + abs(500.0 * (soc - p['SOC_max'])), {
                                "status": "fail_storage_overcharge", "which": "BESS",
                                "delta": float(soc - p['SOC_max'])
                            }
                        if soc < p['SOC_min']:
                            return 1e6 + abs(500.0 * (p['SOC_min'] - soc)), {
                                "status": "fail_storage_overdischarge", "which": "BESS",
                                "delta": float(p['SOC_min'] - soc)
                            }
    
                    # 年累计指标
                    annual_metrics["bess_throughput"] += (p_ch + p_dis) * days_this_type
                    annual_bess_om_cost += (p_ch + p_dis) * p.get('C_BESS_OM_var', 0.0) * days_this_type
    
                    if 'C_FC_OM_var' in p or 'C_EC_OM_var' in p:
                        annual_hss_om_cost += ((p_fc * p.get('C_FC_OM_var', 0.0)) +
                                               (p_ec * p.get('C_EC_OM_var', 0.0))) * days_this_type
                    else:
                        annual_hss_om_cost += (p_fc * p.get('C_HSS_OM_var', 0.0)) * days_this_type
    
                    # === FC degradation ===
                    Δchg  = self.params["fc_deg"]["s_chg"]  * abs(p_fc - p_fc_prev)
                    Δlow  = self.params["fc_deg"]["s_low"]  if (p_fc < 0.2 * n_FC * p['single_fc_rated_power'] and p_fc > 0) else 0
                    Δhigh = self.params["fc_deg"]["s_high"] if (p_fc > 0.8 * n_FC * p['single_fc_rated_power']) else 0
                    Δss   = self.params["fc_deg"]["s_ss"]   if (p_fc > 0 and p_fc_prev == 0) else 0
    
                    annual_metrics["fc_fade"] += (Δchg + Δlow + Δhigh + Δss) * days_this_type
    
                    # === EC degradation ===
                    Δop_ec = self.params["ec_deg"]["s_op"] if p_ec > 0 else 0
                    Δss_ec = self.params["ec_deg"]["s_ss"] if (p_ec > 0 and p_ec_prev == 0) else 0
    
                    annual_metrics["ec_fade"] += (Δop_ec + Δss_ec) * days_this_type
    
                    # === Battery Calendar ===
                    soc_av = 0.5 * (soc_prev + soc)
                    Deg_cal = (
                        self.params["bat_deg"]["k1"] * soc_av**2 +
                        self.params["bat_deg"]["k2"] * soc_av +
                        self.params["bat_deg"]["k3"]
                    )
                    annual_metrics["bess_cal_deg"] += Deg_cal * days_this_type
    
                    # === Battery Cycle ===
                    direction = 1 if p_ch > 0 else (-1 if p_dis > 0 else 0)
    
                    # direction==0 的时候也应记录 SOC 变化
                    if direction == 0:
                        soc_cycle_list.append(soc)
                    else:
                        soc_cycle_list.append(soc)
                        if prev_dir != 0 and direction != prev_dir:
                            if len(soc_cycle_list) > 1:
                                DOD = max(soc_cycle_list) - min(soc_cycle_list)
                                Deg_cyc = (
                                    self.params["bat_deg"]["k4"] * DOD**2 +
                                    self.params["bat_deg"]["k5"] * DOD
                                )
                                annual_metrics["bess_cyc_deg"] += Deg_cyc * days_this_type
                            soc_cycle_list = []
    
                    prev_dir = direction
                    p_fc_prev = p_fc
                    p_ec_prev = p_ec
    
                # —— 期末硬约束 —— 
                tol_soc = p.get("terminal_tol_soc", 1e-6)
                tol_h2  = p.get("terminal_tol_h2",  1e-9)
    
                if n_BESS > 0 and (soc > p['SOC_max'] + tol_soc):
                    return 1e6 + 1e4 * float(soc - p['SOC_max']), {"status": "fail_storage_overcharge", "which": "BESS","delta": float(soc - p['SOC_max'])}
    
                if current_h_storage_kg > max_h_storage_kg + tol_h2:
                    return 1e6 + 1e4 * float(current_h_storage_kg - max_h_storage_kg), {"status": "fail_storage_overcharge", "which": "H2","delta": float(current_h_storage_kg - max_h_storage_kg)}
    
                if n_BESS > 0 and (soc + 1e-12) < (initial_soc - tol_soc):
                    return 1e6 + 1e4 * float(initial_soc - soc), {"status": "fail_storage_overdischarge", "which": "BESS","delta": float(initial_soc - soc)}
    
                if (current_h_storage_kg + 1e-12) < (initial_h_storage_kg - tol_h2):
                    return 1e6 + 1e4 * float(initial_h_storage_kg - current_h_storage_kg), {"status": "fail_storage_overdischarge", "which": "H2","delta": float(initial_h_storage_kg - current_h_storage_kg)}
    
            # 年度 LCOE
            replacement_costs_pv = self._calculate_replacement_costs(solution, annual_metrics)
            total_var_om = annual_hss_om_cost + annual_bess_om_cost
            lcoe = self._calculate_lcoe(solution, total_var_om, annual_buy_back_cost, replacement_costs_pv)
    
            diagnostics["delta_E"] = (P_RES_t.sum() - P_load_t.sum()) / P_load_t.sum() if P_load_t.sum() > 0 else 0.0
            return lcoe, diagnostics
    
        except Exception:
            return 1e6 + 99999, {"status": "simulation_error"}



    def _calculate_pv_power(self, df, n_PV):
        p = self.params
        irradiance, ambient_temp = df['GHI'].values, df['Tair'].values
        rated_power_total = n_PV * p['single_pv_rated_power']
        cell_temp = ambient_temp + ((p['NOCT'] - 20) / 800) * irradiance
        power_output = rated_power_total * p['f_pv'] * (irradiance / p['G_ref']) * (1 + p['alpha_P'] * (cell_temp - p['T_ref']))
        return np.maximum(0, power_output)

    def _calculate_wind_power(self, df, n_WIND):
        p = self.params
        total_wind_rated_power = n_WIND * p['single_wind_rated_power']
        v_ref = df['Vwind'].values
        v_hub = v_ref * (p['wind_hub_height'] / p['wind_ref_height']) ** p['wind_alpha']
        power_output = np.zeros_like(v_hub)
        mask = (v_hub >= p['wind_cut_in_speed']) & (v_hub < p['wind_cut_out_speed'])
        linear_mask, rated_mask = mask & (v_hub < p['wind_rated_speed']), mask & (v_hub >= p['wind_rated_speed'])
        power_output[linear_mask] = total_wind_rated_power * ((v_hub[linear_mask] - p['wind_cut_in_speed']) / (p['wind_rated_speed'] - p['wind_cut_in_speed']))
        power_output[rated_mask] = total_wind_rated_power
        return np.maximum(0, power_output)

    def _calculate_so_hourly_eff_fade(self, p_op, p_rated, d0, gamma, T_op):
        if p_op <= 0 or p_rated <= 0: return 0
        p_deg = self.params["SO_degradation"]
        k_T = np.exp(p_deg["Ea"] / p_deg["R"] * (1/p_deg["T_ref"] - 1/T_op))
        k_L = (p_op / p_rated)**gamma
        return d0 * k_T * k_L

    def _calculate_replacement_costs(self, solution, annual_metrics):
        n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in solution)
        p, r, T = self.params, self.params["discount_rate"], self.params["lifetime"]
        total_pv_replacement_cost = 0.0
        if n_BESS > 0 and annual_metrics["bess_throughput"] > 0:
            total_bess_capacity_kwh = n_BESS * p["C_BESS_initial"]
            if total_bess_capacity_kwh > 0:
                annual_fade_percent = (annual_metrics["bess_throughput"] / total_bess_capacity_kwh) * p["BESS_fade_per_throughput"]
                if annual_fade_percent > 1e-6:
                    years_to_replacement = (1.0 - p["BESS_EoL_SOH"]) / annual_fade_percent
                    if years_to_replacement < T:
                        num_replacements, cost = int(np.floor(T/years_to_replacement)), n_BESS * p["C_BESS_initial"] * p["C_BESS_cap"]
                        for i in range(1, num_replacements + 1): total_pv_replacement_cost += cost / ((1 + r) ** (i * years_to_replacement))
        if n_FC > 0 and annual_metrics["fc_fade"] > 0:
            years_to_replacement = (1.0 - p["SO_EoL_eff_factor"]) / annual_metrics["fc_fade"]
            if years_to_replacement < T:
                num_replacements, cost = int(np.floor(T / years_to_replacement)), n_FC * p["single_fc_rated_power"] * p["C_FC_cap"]
                for i in range(1, num_replacements + 1): total_pv_replacement_cost += cost / ((1 + r) ** (i * years_to_replacement))
        if n_EC > 0 and annual_metrics["ec_fade"] > 0:
            years_to_replacement = (1.0 - p["SO_EoL_eff_factor"]) / annual_metrics["ec_fade"]
            if years_to_replacement < T:
                num_replacements, cost = int(np.floor(T/years_to_replacement)), n_EC * p["single_ec_rated_power"] * p["C_EC_cap"]
                for i in range(1, num_replacements + 1): total_pv_replacement_cost += cost / ((1 + r) ** (i * years_to_replacement))
        return total_pv_replacement_cost

    def _calculate_lcoe(self, solution, total_variable_om_cost_annual, annual_buy_back_cost, pv_replacement_costs):
        n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in solution)
        p, r, T = self.params, self.params["discount_rate"], self.params["lifetime"]
        crf = (r * (1 + r)**T) / ((1 + r)**T - 1)
        initial_cap_cost = (n_PV * p["single_pv_rated_power"] * p["C_RES_cap"] + n_WIND * p["single_wind_rated_power"] * p["C_WIND_cap"] + n_EC * p["single_ec_rated_power"] * p["C_EC_cap"] + n_FC * p["single_fc_rated_power"] * p["C_FC_cap"] + n_TANK * p["C_TANK_cap"] + n_BESS * p["C_BESS_initial"] * p["C_BESS_cap"])
        annual_investment_cost = (initial_cap_cost + pv_replacement_costs) * crf
        om_pv   = n_PV   * p["single_pv_rated_power"]   * p["C_RES_OM"]
        om_wind = n_WIND * p["single_wind_rated_power"] * p["C_WIND_OM"]
        om_bess = n_BESS * p["C_BESS_initial"]          * p["C_BESS_OM"]
        
        # 新增 FC/EC 固定 O&M（若未配置则记 0）
        om_fc   = n_FC   * p["single_fc_rated_power"]   * p.get("C_FC_OM", 0.0)
        om_ec   = n_EC   * p["single_ec_rated_power"]   * p.get("C_EC_OM", 0.0)
        
        annual_fixed_om_cost = om_pv + om_wind + om_bess + om_fc + om_ec

        # total_annual_cost = annual_investment_cost + annual_fixed_om_cost + total_variable_om_cost_annual + annual_buy_back_cost
        total_annual_cost = annual_investment_cost + annual_fixed_om_cost + total_variable_om_cost_annual 
        total_annual_load = sum(df['Load'].sum() * self.day_distribution[i] for i, df in enumerate(self.typical_days_data))
        return total_annual_cost / total_annual_load if total_annual_load > 0 else float('inf')
    
    def simulate_day_trace(self, solution, day_index=0, stochastic=False):
        """
        返回单个典型日的24点时序 DataFrame：
        列：t, P_RES, Load, p_ec, p_fc, p_ch, p_dis, SOC, H2, SOC_raw, H2_raw, status, margin, which
        - SOC/H2 为可视化夹紧值；SOC_raw/H2_raw 为真实未夹紧值
        - 末端按规则标注：>额定上限→overcharge；<期初→overdischarge（仅标注，不中断）
        """
        import numpy as np
        import pandas as pd
    
        p = self.params
        n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in solution)
        # if n_PV < 1 and n_WIND < 1: n_PV = 1
        # if n_EC < 1: n_EC = 1
        # if n_FC < 1: n_FC = 1
        # if n_TANK < 1: n_TANK = 1
    
        day_df = self.typical_days_data[day_index].copy()
        if stochastic and self.stochastic_params.get("enabled", False):
            s = self.stochastic_params
            day_df['GHI']   = np.maximum(0, np.random.normal(day_df['GHI'],   day_df['GHI']   * s['ghi_std_dev_percent']))
            day_df['Vwind'] = np.maximum(0, np.random.normal(day_df['Vwind'], day_df['Vwind'] * s['wind_std_dev_percent']))
            day_df['Load']  = np.maximum(0, np.random.normal(day_df['Load'],  day_df['Load']  * s['load_std_dev_percent']))
    
        dt_hours = 1.0
        total_tank = n_TANK * p['single_tank_capacity_kg']
        hmin, hmax = total_tank * p['SOH_min'], total_tank * p['SOH_max']
        E_bess = n_BESS * p["C_BESS_initial"] if (n_BESS > 0 and p.get("C_BESS_initial", 0) > 0) else 0.0
    
        P_PV_t   = self._calculate_pv_power(day_df, n_PV)
        P_WIND_t = self._calculate_wind_power(day_df, n_WIND)
        P_RES_t  = P_PV_t + P_WIND_t
        P_load_t = day_df['Load'].values
    
        # 期初（与评估一致）
        soc0 = 0.5 * (p['SOC_min'] + p['SOC_max'])
        h0   = 0.5 * (hmin + hmax)
    
        soc     = soc0
        h_store = h0
    
        rows = []
        for t in range(24):
            net = float(P_RES_t[t] - P_load_t[t])
            p_ec = p_fc = p_ch = p_dis = 0.0
            status = "success"; margin = 0.0; which = ""
    
            if net > 0:
                p_ec = min(net, n_EC * p['single_ec_rated_power'])
                leftover = net - p_ec
                p_ch = min(leftover, n_BESS * p['P_ch_rated']) if n_BESS > 0 else 0.0
            else:
                demand = -net
                p_fc  = min(demand, n_FC * p['single_fc_rated_power'])
                leftover = demand - p_fc
                p_dis = min(leftover, n_BESS * p['P_dis_rated']) if n_BESS > 0 else 0.0
                rem = demand - p_fc - p_dis
                if rem > 1e-2:
                    status = "fail_supply"; margin = float(rem); which = ""
    
            # 状态更新（raw）
            if p['HHV'] > 0:
                h_prod = (p_ec * p['eta_EC_initial']) / p['HHV']
                h_cons = (p_fc / p['eta_FC_initial']) / p['HHV'] if p['eta_FC_initial'] > 0 else 0.0
            else:
                h_prod = 0.0; h_cons = 0.0
            h_store = h_store + h_prod - h_cons
    
            if E_bess > 0:
                soc = soc + (p_ch * p['eta_ch'] - p_dis / p['eta_dis']) * dt_hours / E_bess
    
            # 中间越界标注（不中断）
            if h_store > hmax:
                status, margin, which = "fail_storage_overcharge", float(h_store - hmax), "H2"
            elif h_store < hmin:
                status, margin, which = "fail_storage_overdischarge", float(hmin - h_store), "H2"
    
            if n_BESS > 0:
                if soc > p['SOC_max']:
                    status, margin, which = "fail_storage_overcharge", float(soc - p['SOC_max']), "BESS"
                elif soc < p['SOC_min']:
                    status, margin, which = "fail_storage_overdischarge", float(p['SOC_min'] - soc), "BESS"
    
            # 可视化夹紧
            soc_vis = min(max(soc,  p['SOC_min']), p['SOC_max'])
            h2_vis  = min(max(h_store, hmin),      hmax)
    
            rows.append({
                "t": t,
                "P_RES": float(P_RES_t[t]),
                "Load":  float(P_load_t[t]),
                "p_ec":  float(p_ec),
                "p_fc":  float(p_fc),
                "p_ch":  float(p_ch),
                "p_dis": float(p_dis),
                "SOC":   float(soc_vis),
                "H2":    float(h2_vis),
                "SOC_raw": float(soc),
                "H2_raw":  float(h_store),
                "status": status,
                "margin": float(margin),
                "which":  which
            })
    
        # —— 期末标注（仅可视化）：>额定上限→overcharge；<期初→overdischarge —— 
        df = pd.DataFrame(rows)
        tol_soc = p.get("terminal_tol_soc", 1e-6)
        tol_h2  = p.get("terminal_tol_h2",  1e-9)
    
        soc_end = float(df.iloc[-1].get("SOC_raw", df.iloc[-1]["SOC"]))
        h2_end  = float(df.iloc[-1].get("H2_raw",  df.iloc[-1]["H2"]))
    
        # 先判超过额定
        if n_BESS > 0 and (soc_end > p['SOC_max'] + tol_soc):
            df.at[df.index[-1], "status"] = "fail_storage_overcharge"
            df.at[df.index[-1], "which"]  = "BESS"
            df.at[df.index[-1], "margin"] = float(soc_end - p['SOC_max'])
    
        if h2_end > hmax + tol_h2:
            df.at[df.index[-1], "status"] = "fail_storage_overcharge"
            df.at[df.index[-1], "which"]  = "H2"
            df.at[df.index[-1], "margin"] = float(h2_end - hmax)
    
        # 再判少于期初
        if n_BESS > 0 and (soc_end + 1e-12) < (soc0 - tol_soc):
            df.at[df.index[-1], "status"] = "fail_storage_overdischarge"
            df.at[df.index[-1], "which"]  = "BESS"
            df.at[df.index[-1], "margin"] = float(soc0 - soc_end)
    
        if (h2_end + 1e-12) < (h0 - tol_h2):
            df.at[df.index[-1], "status"] = "fail_storage_overdischarge"
            df.at[df.index[-1], "which"]  = "H2"
            df.at[df.index[-1], "margin"] = float(h0 - h2_end)
    
        return df
