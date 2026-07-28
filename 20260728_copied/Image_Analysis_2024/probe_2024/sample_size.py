from statsmodels.stats.power import FTestAnovaPower
import numpy as np
from scipy.stats import linregress
from scipy.stats import kruskal
from sklearn.utils import resample

# グループ1の時系列データ
group1 = [
    [138, 154, 160, 150],  # サンプル1
    [105, 157, 162, 171],  # サンプル2
    [50, 155, 154, 171],  # サンプル3
    [80, 117, 159, 156],
    [96, 134, 137, 147],
    [110, 153, 140, 189],
    [154, 150, 187, 185],
    [127, 175, 164, 174],
    [120, 131, 158, 169],
    [161, 196, 202, 184],
    [122, 147, 167, 193],
    [115, 154, 120, 161],
    [98, 129, 124, 144],
    [107, 167, 160, 141],
    [143, 197, 165, 195]
]

group2 = [
    [75, 117, 89, 129],
    [80, 93, 100, 106],
    [82, 122, 98, 113],
    [85, 114, 133, 126],
    [142, 154, 147, 158],
    [177, 153, 138, 155],
    [124, 83, 149, 181],
    [115, 131, 142, 89]
]

group3 = [
    [146, 170, 169, 147],
    [75, 111, 118, 167],
    [166, 164, 171, 169]
]
time_points1 = np.array(group1).T
time_points2 = np.array(group2).T
time_points3 = np.array(group3).T

slopes_each1 = [linregress(range(len(sample)), sample).slope for sample in time_points1]
slopes_each2 = [linregress(range(len(sample)), sample).slope for sample in time_points2]
slopes_each3 = [linregress(range(len(sample)), sample).slope for sample in time_points3]

print(f"各サンプルの傾き_slopes_each1: {slopes_each1}")
print(f"各サンプルの傾き_slopes_each2: {slopes_each2}")
print(f"各サンプルの傾き_slopes_each3: {slopes_each3}")

# 各時点でのグループ平均値を計算
 # 転置して時点ごとにアクセス
group1_means = [np.mean(tp) for tp in time_points1]
  # 転置して時点ごとにアクセス
group2_means = [np.mean(tp) for tp in time_points2]
  # 転置して時点ごとにアクセス
group3_means = [np.mean(tp) for tp in time_points3]
# 平均値に基づいて傾きを計算
slope1, intercept1, r_value1, p_value1, stderr1 = linregress(range(len(group1_means)), group1_means)
slope2, intercept2, r_value2, p_value2, stderr2 = linregress(range(len(group2_means)), group2_means)
slope3, intercept3, r_value3, p_value3, stderr3 = linregress(range(len(group3_means)), group3_means)

print(f"グループ1の傾き: {slope1}, r: {r_value1}, p: {p_value1}")
print(f"グループ2の傾き: {slope2}, r: {r_value2}, p: {p_value2}")
print(f"グループ3の傾き: {slope3}, r: {r_value3}, p: {p_value3}")
print(f"group1_means: {group1_means}")
print(f"group2_means: {group2_means}")
print(f"group3_means: {group3_means}")
print(f"time_points1: {time_points1}")
print(f"time_points2: {time_points2}")
print(f"time_points3: {time_points3}")

group_means = [slope1, slope2, slope3]
group_ns = [len(group1), len(group2), len(group3)]
overall_mean = np.mean(group_means)

# 群間平方和 (SSB)
ssb = sum(n * (mean - overall_mean)**2 for n, mean in zip(group_ns, group_means))

# 群内平方和 (SSW)
ssw = sum(
    sum((slope - mean)**2 for slope in group_slopes)
    for group_slopes, mean in zip([slopes_each1, slopes_each2, slopes_each3], group_means)
)

# 全平方和 (SST)
sst = ssb + ssw

# 効果量 (eta^2 と f^2)
eta_squared = ssb / sst
cohen_f_squared = eta_squared / (1 - eta_squared)

print(f"群間平方和 (SSB): {ssb:.2f}")
print(f"群内平方和 (SSW): {ssw:.2f}")
print(f"全平方和 (SST): {sst:.2f}")
print(f"効果量 eta^2: {eta_squared:.3f}")
print(f"効果量 f^2: {cohen_f_squared:.3f}")

time_points4 = resample(time_points1)



def simulate_power_with_effect_size(time_points1, time_points2, time_points3, time_points4, n_iter=1000, alpha=0.05):
    significant_count = 0
    H_values = []  # H統計量を記録するリスト

    for _ in range(n_iter):
        resampled1 = resample(time_points1, n_samples=len(time_points1), replace=True).flatten()
        resampled2 = resample(time_points2, n_samples=len(time_points2), replace=True).flatten()
        resampled3 = resample(time_points3, n_samples=len(time_points3), replace=True).flatten()
        resampled4 = resample(time_points4, n_samples=len(time_points4), replace=True).flatten()

        # Kruskal-Wallis検定
        H, p = kruskal(resampled1, resampled2, resampled3, resampled4)

        # 統計量Hを記録
        H_values.append(H)

        # p値が有意水準以下ならカウント
        if np.isscalar(p) and p < alpha:
            significant_count += 1

    # 平均的なH統計量を使用して効果量を計算
    mean_H = np.mean(H_values)
    N = len(time_points1.flatten()) + len(time_points2.flatten()) + len(time_points3.flatten()) + len(time_points4.flatten())
    eta_squared = mean_H / (N - 1)  # Kruskal-Wallisの効果量 η²

    power = significant_count / n_iter

    return power, eta_squared

# シミュレーション実行
power_005, eta_squared_005 = simulate_power_with_effect_size(time_points1, time_points2, time_points3, time_points4, n_iter=500, alpha=0.05)
power_001, eta_squared_001 = simulate_power_with_effect_size(time_points1, time_points2, time_points3, time_points4, n_iter=500, alpha=0.01)

# 結果表示
print(f"4群の検出力 (alpha=0.05): {power_005:.2f}")
print(f"効果量 η² (alpha=0.05): {eta_squared_005:.3f}")
print(f"4群の検出力 (alpha=0.01): {power_001:.2f}")
print(f"効果量 η² (alpha=0.01): {eta_squared_001:.3f}")

 # パラメータ設定
cohen_f = cohen_f_squared**0.5

alpha = 0.05        # 有意水準
power = 0.8         # 検出力
k_groups = 4        # グループ数

# サンプルサイズ計算
power_analysis = FTestAnovaPower()
sample_size = power_analysis.solve_power(effect_size=cohen_f, alpha=alpha, power=power, k_groups=k_groups)

print(f"各グループに必要なサンプルサイズ: {sample_size:.2f}")


def estimate_sample_size(time_points1, time_points2, time_points3, time_points4, alpha=0.01, target_power=0.8, max_sample_size=500,
                         n_iter=1000):
    """
    シミュレーションを用いて必要なサンプルサイズを推定する。

    Parameters:
        time_points1, time_points2, time_points3: ndarray
            元の時系列データ。
        alpha: float
            有意水準。
        target_power: float
            目標検出力。
        max_sample_size: int
            最大サンプルサイズの探索範囲。
        n_iter: int
            シミュレーション回数。

    Returns:
        min_sample_size: int
            目標検出力を達成する最小のサンプルサイズ。
    """
    from scipy.stats import kruskal

    for sample_size in range(5, max_sample_size + 1, 5):  # 5刻みでサンプルサイズを増やす
        significant_count = 0
        for _ in range(n_iter):
            resampled1 = resample(time_points1, n_samples=sample_size, replace=True).flatten()
            resampled2 = resample(time_points2, n_samples=sample_size, replace=True).flatten()
            resampled3 = resample(time_points3, n_samples=sample_size, replace=True).flatten()
            resampled4 = resample(time_points4, n_samples=sample_size, replace=True).flatten()
            # Kruskal-Wallis検定
            _, p = kruskal(resampled1, resampled2, resampled3, resampled4)

            # p値が有意水準以下ならカウント
            if p < alpha:
                significant_count += 1

        # 検出力を計算
        power = significant_count / n_iter

        # 目標検出力に達したら、そのサンプルサイズを返す
        if power >= target_power:
            return sample_size

    # 最大サンプルサイズまで達しても目標に到達しない場合
    return None


# 必要なサンプルサイズを推定
estimated_sample_size = estimate_sample_size(
    time_points1, time_points2, time_points3, time_points4, alpha=0.01, target_power=0.8, max_sample_size=500, n_iter=500
)

print(f"目標検出力を達成する最小のサンプルサイズ: {estimated_sample_size}")



def estimate_sample_size_with_eta_squared(eta_squared, alpha=0.05, target_power=0.8, max_sample_size=500, k_groups=4):
    """
    効果量 η^2 を用いて、目標検出力を達成するための最小サンプルサイズを推定する。

    Parameters:
        eta_squared: float
            効果量 η^2。
        alpha: float
            有意水準。
        target_power: float
            目標検出力。
        max_sample_size: int
            最大サンプルサイズの探索範囲。
        k_groups: int
            グループ数。

    Returns:
        min_sample_size: int
            目標検出力を達成する最小のサンプルサイズ。
    """
    # η² から Cohen's f を計算
    cohen_f = np.sqrt(eta_squared / (1 - eta_squared))

    # サンプルサイズ推定
    power_analysis = FTestAnovaPower()
    for sample_size in range(2, max_sample_size + 1):  # サンプルサイズを2から探索
        power = power_analysis.solve_power(effect_size=cohen_f, nobs=sample_size * k_groups,
                                           alpha=alpha, k_groups=k_groups)
        if power >= target_power:
            return sample_size
    return None  # 最大サンプルサイズまでに目標検出力が達成されない場合

# 効果量 η² = 0.123、目標検出力 0.8、最大サンプルサイズ 500、グループ数 4 で推定
eta_squared = 0.305
alpha = 0.01
target_power = 0.8
max_sample_size = 500
k_groups = 4

min_sample_size = estimate_sample_size_with_eta_squared(eta_squared, alpha, target_power, max_sample_size, k_groups)

if min_sample_size:
    print(f"目標検出力を達成する最小のサンプルサイズ: {min_sample_size}")
else:
    print("最大サンプルサイズ内で目標検出力に達することができませんでした。")