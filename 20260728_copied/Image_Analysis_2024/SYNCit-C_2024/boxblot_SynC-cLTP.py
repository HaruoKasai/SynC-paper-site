import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
from matplotlib.backends.backend_pdf import PdfPages

dir_name = r"X:\SYNCit-C\SynC_Fig1\Fig.1_e-f\_summary_by_dend"  # 输入数据文件夹路径
csv_list = glob.glob(os.path.join(dir_name, "*.csv"))

# 用硬编码切掉固定长度，得到组名
cond_list = sorted(list(set([s[:-16] for s in csv_list])))
cond_num = len(cond_list)
print(f"找到 {cond_num} 个组：", cond_list)

tp_after = [6, 8]   # Python 索引 → 第8~9行
channel_results = {"c1": {}, "c2": {}}

# 遍历每个组
for cond in cond_list:
    c_list = sorted(glob.glob(cond + "_c*"))
    if len(c_list) < 3:
        print(f"⚠️ {cond} 的文件数不足3个，跳过")
        continue

    # 分别读取 c0, c1, c2
    df_c0 = pd.read_csv(c_list[0]).iloc[tp_after[0]:tp_after[1], 1:-3]
    df_c1 = pd.read_csv(c_list[1]).iloc[tp_after[0]:tp_after[1], 1:-3]
    df_c2 = pd.read_csv(c_list[2]).iloc[tp_after[0]:tp_after[1], 1:-3]

    # flatten 数据
    values_c1 = df_c1.values.flatten().astype(float)
    values_c2 = df_c2.values.flatten().astype(float)

    group = os.path.basename(cond)  # 组名
    channel_results["c1"][group] = values_c1
    channel_results["c2"][group] = values_c2

# 创建保存目录
save_dir = dir_name + "_sum_bar"
os.makedirs(save_dir, exist_ok=True)

# 保存数据 & 作图函数
def save_and_plot(channel, results, pdf):
    if not results:
        print(f"⚠️ {channel} 没有数据，跳过绘图")
        return

    # 保存原始数据表
    out_df = pd.DataFrame(dict([(g, pd.Series(v)) for g, v in results.items()]))
    out_path = os.path.join(save_dir, f"{channel}_raw_values.csv")
    out_df.to_csv(out_path, index=False)

    # 计算统计指标
    stats = []
    for g, v in results.items():
        v = pd.Series(v).dropna()
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        iqr = q3 - q1
        whisker_low = max(v.min(), q1 - 1.5*iqr)
        whisker_high = min(v.max(), q3 + 1.5*iqr)
        stats.append({
            "group": g,
            "N": v.count(),
            "mean": v.mean(),
            "std": v.std(),
            "median": v.median(),
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "whisker_low": whisker_low,
            "whisker_high": whisker_high
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(save_dir, f"{channel}_summary_stats.csv"), index=False)


    # Kruskal-Wallis test
    H, p = kruskal(*results.values())
    with open(os.path.join(save_dir, f"{channel}_kruskal.txt"), "w") as f:
        f.write(f"Kruskal-Wallis test: H = {H:.3f}, p = {p:.3e}\n")

    # 画图
    df_long = pd.DataFrame({
        "group": np.concatenate([[g] * len(v) for g, v in results.items()]),
        "value": np.concatenate([v for v in results.values()])
    })

    # Dunn’s test (Bonferroni correction)
    dunn = sp.posthoc_dunn(df_long, val_col="value", group_col="group") # p_adjust='bonferroni'
    dunn.to_csv(os.path.join(save_dir, f"{channel}_dunn_posthoc.csv"))




    plt.figure(figsize=(5,6))
    sns.boxplot(x="group", y="value", data=df_long, palette="Set2")
    sns.stripplot(x="group", y="value", data=df_long, color="black", alpha=0.7, jitter=0.2)
    plt.ylabel(f"Δ{channel.upper()} (%)")
    plt.ylim(-55, 160)
    plt.yticks(np.arange(-50, 155, 50))
    plt.xlabel("")
    plt.title(f"{channel.upper()} Delta (%)\nKruskal-Wallis p={p:.1e}")
    plt.tight_layout()
    pdf.savefig()  # 保存到同一个pdf
    plt.axes()
    plt.close()

# 合并 PDF 输出
pdf_path = os.path.join(save_dir, "c1_c2_boxplots.pdf")
with PdfPages(pdf_path) as pdf:
    save_and_plot("c1", channel_results["c1"], pdf)
    save_and_plot("c2", channel_results["c2"], pdf)

print(f"✅ 已完成，结果保存在: {save_dir}")
