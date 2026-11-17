# utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_statistical_results(histories_list, labels, 
                             csv_path="convergence_results.csv",
                             pdf_path="convergence_plot.pdf"):
    """
    绘制统计收敛曲线（均值 ± 标准差），保存为 PDF（矢量图）和 CSV。
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    from matplotlib import rcParams

    # 论文格式
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 20

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.grid(False)


    colors = ['royalblue', 'coral', 'mediumseagreen']
    csv_dict = {"iteration": None}

    for i, histories in enumerate(histories_list):
        if not histories:
            continue

        # 过滤掉完全空的 run
        histories = [h for h in histories if len(h) > 0]
        if not histories:
            continue

        max_len = max(len(h) for h in histories)
        padded = []

        for h in histories:
            h_clean = [x if x < 1e6 else np.nan for x in h]
            h_pad = h_clean + [np.nan] * (max_len - len(h_clean))
            padded.append(h_pad)

        padded_arr = np.array(padded, dtype=float)
        # 如果这一组全是 NaN，就跳过，避免 warning
        if np.all(np.isnan(padded_arr)):
            continue

        mean_hist = np.nanmean(padded_arr, axis=0)
        std_hist  = np.nanstd(padded_arr, axis=0)
        iterations = np.arange(max_len)

        ax.plot(
            iterations, mean_hist,
            label=labels[i],
            color=colors[i % len(colors)],
            linewidth=1.5
        )
        ax.fill_between(
            iterations,
            mean_hist - std_hist,
            mean_hist + std_hist,
            color=colors[i % len(colors)],
            alpha=0.20
        )

        csv_dict["iteration"] = iterations
        csv_dict[f"{labels[i]}_mean"] = mean_hist
        csv_dict[f"{labels[i]}_std"]  = std_hist

    # 坐标轴和图例
    ax.set_xlabel('Iteration')
    ax.set_ylabel('LCOE ($/kWh)')
    ax.tick_params(axis='both')
    ax.legend(frameon=False)
    # ax.set_yscale('log')

    fig.tight_layout()

    # 先保存 PDF 再 show
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")

    # 保存 CSV
    # df = pd.DataFrame(csv_dict)
    # df.to_csv(csv_path, index=False)
    # print(f"CSV saved to: {csv_path}")

    plt.show()
    plt.close(fig)


def plot_final_fitness_boxplot(final_fitness_data, title="Final Fitness Distribution Comparison"):
    """
    绘制最终适应度分布的箱形图。
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    filtered_data = {}
    for key, values in final_fitness_data.items():
        filtered_values = [v for v in values if v < 1e6]
        if filtered_values:
            filtered_data[key] = filtered_values

    if not filtered_data:
        print("Warning: No valid final fitness data to plot in boxplot.")
        return
        
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in filtered_data.items() ]))
    
    sns.boxplot(data=df, ax=ax, palette="Set2")
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel('Final Best Fitness (LCOE)', fontsize=12)
    ax.set_xlabel('Algorithm', fontsize=12)
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()

# --- utils.py 追加 ---
import csv, os, numpy as np, pandas as pd
class ExperimentLogger:
    def __init__(self, outdir="runs/ga_sac"):
        os.makedirs(outdir, exist_ok=True)
        self.meta_path = os.path.join(outdir, "iter_log.csv")
        self.trace_dir = os.path.join(outdir, "traces"); os.makedirs(self.trace_dir, exist_ok=True)
        self._init_csv()

    def _init_csv(self):
        self.fieldnames = [
            "iter","treat_prob","gamma_fb",
            "gbest","curr_best","reward",
            "iter_ratio","diversity","convergence_rate","gbest_status",
            "cnt_success","cnt_fail_supply","cnt_fail_overdis","cnt_fail_overchg","pop_size"
        ]
        with open(self.meta_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    @staticmethod
    def _pop_diversity(pop):
        arr = np.array(pop)
        return float(np.std(arr, axis=0).mean())

    @staticmethod
    def _status_counts(ev_res):
        mp = {"success":0, "fail_supply":0, "fail_storage_overdischarge":0, "fail_storage_overcharge":0}
        for r in ev_res:
            mp[r["diagnostics"].get("status","success")] = mp.get(r["diagnostics"].get("status","success"),0)+1
        return mp

    def log_iter(self, iteration, total_iterations, action_vec, gbest, curr_best, reward, ev_res, gbest_diag, conv_rate):
        treat_prob = float(action_vec[0]); gamma_fb = float(action_vec[1])
        status = (gbest_diag or {}).get("status","success")
        sc = self._status_counts(ev_res)
        rec = {
            "iter": iteration,
            "treat_prob": treat_prob,
            "gamma_fb": gamma_fb,
            "gbest": gbest, "curr_best": curr_best, "reward": reward,
            "iter_ratio": iteration/total_iterations,
            "diversity": self._pop_diversity([r["individual"] for r in ev_res]),
            "convergence_rate": conv_rate,
            "gbest_status": status,
            "cnt_success": sc.get("success",0),
            "cnt_fail_supply": sc.get("fail_supply",0),
            "cnt_fail_overdis": sc.get("fail_storage_overdischarge",0),
            "cnt_fail_overchg": sc.get("fail_storage_overcharge",0),
            "pop_size": len(ev_res)
        }
        import csv
        with open(self.meta_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(rec)

    def save_trace_df(self, df, tag):
        p = os.path.join(self.trace_dir, f"trace_{tag}.csv")
        df.to_csv(p, index=False)
        return p
    
import os
import csv
import json
import numpy as np

def decode_solution_to_config(solution):
    """
    把连续解向量解码为整数设备配置（负数→0，正数→ceil）
    返回: dict
    """
    n_PV, n_WIND, n_EC, n_FC, n_TANK, n_BESS = (int(np.ceil(x)) if x > 0 else 0 for x in solution)
    # 可选：防止全零导致奇异
    if n_PV < 1 and n_WIND < 1: n_PV = 1
    if n_EC  < 1: n_EC = 1
    if n_FC  < 1: n_FC = 1
    if n_TANK< 1: n_TANK = 1
    return {
        "n_PV": n_PV, "n_WIND": n_WIND, "n_EC": n_EC,
        "n_FC": n_FC, "n_TANK": n_TANK, "n_BESS": n_BESS
    }

class BestConfigWriter:
    """
    轻量写手：把每种算法的最终最优解配置记录到 runs/xxx/best_configs.csv
    """
    def __init__(self, outdir):
        self.path = os.path.join(outdir, "best_configs.csv")
        # 如果不存在，写表头
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "algorithm", "seed", "status", "LCOE",
                    "n_PV","n_WIND","n_EC","n_FC","n_TANK","n_BESS",
                    "solution_json"
                ])

    def append(self, algorithm, seed, status, lcoe, solution):
        cfg = decode_solution_to_config(solution)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                algorithm, seed, status, float(lcoe),
                cfg["n_PV"], cfg["n_WIND"], cfg["n_EC"], cfg["n_FC"], cfg["n_TANK"], cfg["n_BESS"],
                json.dumps([float(x) for x in solution], ensure_ascii=False)
            ])
        print(f"[best-config] {algorithm}/seed={seed} → LCOE={lcoe:.6f}, status={status}, cfg={cfg}")

