import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ================== 读数据函数 ==================
def load_typical_days_data(data_folder="data"):
    typical_days_data = []
    required_columns = ['GHI', 'Tair', 'Vwind', 'Load']
    for i in range(5):
        fp = os.path.join(data_folder, f"typicalday{i}.csv")
        if not os.path.exists(fp):
            print(f"[ERROR] not found: {fp}")
            return None
        df = pd.read_csv(fp)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            print(f"[ERROR] missing columns in {fp}: {missing}")
            return None
        typical_days_data.append(df)
    print(f"[INFO] loaded {len(typical_days_data)} typical days from '{data_folder}'")
    return typical_days_data


# ================== 画图函数 ==================
def plot_typical_days_curves(
    data_folder="data",
    out_folder="figs_typical_days"
):
    mpl.rcParams['font.family'] = 'Times New Roman'

    typical_days_data = load_typical_days_data(data_folder=data_folder)
    if typical_days_data is None:
        return

    os.makedirs(out_folder, exist_ok=True)

    vars_to_plot = ['GHI', 'Tair', 'Vwind', 'Load']

    # 变量对应的 ylabel
    ylabels = {
        'GHI': "GHI (W/m$^2$)",
        'Tair': "Temperature (℃)",
        'Vwind': "Wind Speed (m/s)",
        'Load': "Load (kW)"
    }

    # 颜色（5 天）
    colors = ['tab:blue', 'tab:orange', 'tab:red', 'tab:grey', 'tab:green']

    for var in vars_to_plot:
        fig, ax = plt.subplots(figsize=(6, 4))


        for day_idx, df in enumerate(typical_days_data):
            x = np.arange(len(df[var]))  # 假设是 24 个或更多小时
            y = df[var].values

            ax.plot(
                x, y,
                linewidth=1.5,
                label=f"Day {day_idx}",
                color=colors[day_idx]
            )

        # 上限加一点 margin 让 legend 不遮挡曲线
        # 在自动 ylim 基础上，仅把上限抬高一点
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.10)

        ax.set_xlabel("Hour (h)")
        ax.set_ylabel(ylabels[var])

        ax.legend(
            loc="upper right",
            ncol=5,
            fontsize=9,
            frameon=False
        )

        ax.grid(False)

        out_path = os.path.join(out_folder, f"{var}_typical_days.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.show(fig)

        print(f"[INFO] saved: {out_path}")


if __name__ == "__main__":
    plot_typical_days_curves(data_folder="data")
