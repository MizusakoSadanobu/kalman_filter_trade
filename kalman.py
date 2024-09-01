import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def data_gen(a, su, sw, T, show_graph):
    # ノイズ生成
    u = np.random.normal(0, su, T)
    u[1:] = a * u[:-1] + np.random.normal(0, su, T - 1)
    w = np.random.normal(0, sw, T)
    w[1:] = w[:-1] + np.random.normal(0, sw, T - 1)

    # トレンドと対数価格の生成
    trend = np.cumsum(w)
    log_price = np.cumsum(trend + u)

    # DataFrameに変換
    df = pd.DataFrame({'log_price': log_price})
    if show_graph:
        df.plot()
        plt.show()
    return df

def pred_confidence_interval(kf, df):
    """
    kalmanフィルタによる信頼区間を求める
    """
    lower95s = []
    upper95s = []
    lower60s = []
    upper60s = []
    trends = []
    for i in range(len(df)):
        # カルマンフィルタの更新（i時刻までのデータでフィルタリング）
        window = 50
        state_means, state_covs = (
            kf.filter(df['log_price'].values[:i+1])
            if i<window else kf.filter(df['log_price'].values[i-window:i+1])
        )
        trends.append(state_means[-1,1])
        
        # 信頼区間の計算
        lower95 = state_means[-1, 0] - 0.1 * np.sqrt(state_covs[-1, 0, 0])
        upper95 = state_means[-1, 0] + 0.1 * np.sqrt(state_covs[-1, 0, 0])
        lower60 = state_means[-1, 0] - 0. * np.sqrt(state_covs[-1, 0, 0])
        upper60 = state_means[-1, 0] + 0. * np.sqrt(state_covs[-1, 0, 0])

        lower95s.append(lower95)
        upper95s.append(upper95)
        lower60s.append(lower60)
        upper60s.append(upper60)

    df["lower95"] = lower95s
    df["upper95"] = upper95s
    df["lower60"] = lower60s
    df["upper60"] = upper60s
    df["trend"] = trends
    return df

def calc_return(df):
    """
    利益を求める
    """
    df["long_entry"] = (df["log_price"] > df["upper95"]) & (df["trend"]>0)
    df["short_entry"] = (df["log_price"] < df["lower95"]) & (df["trend"]<0)

    df["long_close"] = df["log_price"] < df["lower60"]
    df["short_close"] = df["log_price"] > df["upper60"]

    df["long_ret"] = 0.
    pos = 0
    for i, (entry, close, ret) in enumerate(zip(df["long_entry"], df["long_close"], df["log_price"])):
        df.loc[i,"long_ret"] = ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0


    df["short_ret"] = 0.
    pos = 0
    for i, (entry, close, ret) in enumerate(zip(df["short_entry"], df["short_close"], df["log_price"])):
        df.loc[i,"short_ret"] = -ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0

    df["both_ret"] = df["short_ret"] + df["long_ret"]

    return df    