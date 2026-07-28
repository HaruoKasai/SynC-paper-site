# lib/steel_test.py

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal


def steel_test(x, g, control=None, alternative="two.sided"):
    """
    Steel's Many-to-One Rank Test (Python implementation)

    Parameters
    ----------
    x : array-like
        Observations (numeric values).
    g : array-like
        Group labels (same length as x).
    control : str or int, optional
        Control group label. Default = first level of g.
    alternative : str, optional
        "two.sided", "less", or "greater".

    Returns
    -------
    results : pandas.DataFrame
        Columns: comparison, t.value, rho, p.value
    """

    x = np.asarray(x)
    g = np.asarray(g)

    if len(x) != len(g):
        raise ValueError("'x' and 'g' must have the same length")

    # 去掉 NaN
    mask = ~np.isnan(x)
    x = x[mask]
    g = g[mask]

    groups, g_idx = np.unique(g, return_inverse=True)
    a = len(groups)
    if a < 2:
        raise ValueError("Need at least 2 groups")

    if control is None:
        control = groups[0]
    if control not in groups:
        raise ValueError("Control group not found in data")

    ni = np.array([np.sum(g == grp) for grp in groups])
    n1 = ni[np.where(groups == control)][0]

    # ρ 的计算（参照 R 代码）
    def get_rho(ni):
        l = len(ni)
        rho_matrix = np.zeros((l, l))
        for i in range(l):
            for j in range(l):
                if i != j:
                    rho_matrix[i, j] = np.sqrt(
                        (ni[i] / (ni[i] + n1)) * (ni[j] / (ni[j] + n1))
                    )
        rho = np.sum(rho_matrix[1:, 1:]) / ((l - 2) * (l - 1))
        return rho

    if np.all(ni == n1):
        rho = 0.5
    else:
        rho = get_rho(ni)

    results = []

    for grp in groups:
        if grp == control:
            continue

        xc = x[g == control]
        xt = x[g == grp]
        n2 = len(xt)

        # ranking
        r = rankdata(np.concatenate([xc, xt]))
        R = np.sum(r[:n1])
        N = n1 + n2

        E = n1 * (N + 1) / 2
        V = n1 * n2 / N / (N - 1) * (np.sum(r**2) - N * (N + 1) ** 2 / 4)

        t_value = (R - E) / np.sqrt(V)

        # correlation matrix
        corr = np.eye(a - 1)
        corr[np.tril_indices(a - 1, -1)] = rho
        corr[np.triu_indices(a - 1, 1)] = rho

        if alternative == "less":
            lower, upper = -t_value, np.inf
        elif alternative == "greater":
            lower, upper = t_value, np.inf
        else:  # two.sided
            t_value = abs(t_value)
            lower, upper = -t_value, t_value

        # 多元正态分布近似 (这里模拟 R 的 mvtnorm::pmvt)
        mean = np.zeros(a - 1)
        dist = multivariate_normal(mean=mean, cov=corr)

        # NOTE: 这里简化处理，实际上 pmvt 是多元 t 分布，这里用 normal approximation
        p_value = 1 - dist.cdf([upper] * (a - 1))

        results.append(
            {
                "comparison": f"{grp}:{control}",
                "t.value": t_value,
                "rho": rho,
                "p.value": p_value,
            }
        )

    return pd.DataFrame(results)
