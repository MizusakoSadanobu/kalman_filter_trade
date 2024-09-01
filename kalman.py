import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def data_gen(a, su, sw, T, show_graph):
    # ノイズ生成
    log_price = np.zeros(T)
    u = np.random.normal(0, su, T)
    for t in range(T):
        if t<=100:
            log_price[t] = 1000
        else:
            log_price[t] = (log_price[t-1]-log_price[t-2])*a + log_price[t-1] + u[t]
    w = np.random.normal(0, sw, T)
    log_price = log_price + w

    # DataFrameに変換
    df = pd.DataFrame({'log_price': log_price})
    if show_graph:
        df.plot()
        plt.show()
    return df

def pred_confidence_interval(df, su, sw):
    """
    kalmanフィルタによる信頼区間を求める
    """
    lower95s = []
    upper95s = []
    lower60s = []
    upper60s = []
    trends = []

    # モデル初期化
    state_means = [0, 0]
    state_covs = [[1, 0], [0, 1]]
    kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]],
                    observation_matrices=[[1, 0]],
                    initial_state_mean=state_means,
                    initial_state_covariance=state_covs,
                    observation_covariance=su,
                    transition_covariance=[[0, 0], [0, sw]])
    
    for i in range(len(df)):
        # カルマンフィルタの更新（i時刻までのデータでフィルタリング）
        state_means, state_covs = kf.filter_update(
            filtered_state_mean = state_means,
            filtered_state_covariance = state_covs,
            observation = [df['log_price'].values[i]],
        )
        trends.append(state_means[1])
        
        # 信頼区間の計算
        lower95 = state_means[0] - 2 * np.sqrt(state_covs[0, 0]+sw)
        upper95 = state_means[0] + 2 * np.sqrt(state_covs[0, 0]+sw)
        lower60 = state_means[0] - 0.5 * np.sqrt(state_covs[0, 0]+sw)
        upper60 = state_means[0] + 0.5 * np.sqrt(state_covs[0, 0]+sw)

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
    for i, (idx, entry, close, ret) in enumerate(zip(df.index, df["long_entry"], df["long_close"], df["log_price"].diff().fillna(0))):
        df.loc[idx,"long_ret"] = ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0


    df["short_ret"] = 0.
    pos = 0
    for i, (idx, entry, close, ret) in enumerate(zip(df.index, df["short_entry"], df["short_close"], df["log_price"].diff().fillna(0))):
        df.loc[idx,"short_ret"] = -ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0

    df["both_ret"] = df["short_ret"] + df["long_ret"]

    return df

def calc_base_return(df):
    df["cci"] = (df["log_price"] - df["log_price"].rolling(30).mean())/(df["log_price"].rolling(30).std())
    df["long_entry"] = (df["cci"] < -2)
    df["short_entry"] = (df["cci"] > 2)

    df["long_close"] = (df["cci"] > 1)
    df["short_close"] = (df["cci"] < -1)

    df["long_ret"] = 0.
    pos = 0
    for i, (idx, entry, close, ret) in enumerate(zip(df.index, df["long_entry"], df["long_close"], df["log_price"].diff().fillna(0))):
        df.loc[idx,"long_ret"] = ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0


    df["short_ret"] = 0.
    pos = 0
    for i, (idx, entry, close, ret) in enumerate(zip(df.index, df["short_entry"], df["short_close"], df["log_price"].diff().fillna(0))):
        df.loc[idx,"short_ret"] = -ret*pos
        if entry:
            pos += 1
        if close:
            pos = 0

    df["both_ret"] = df["short_ret"] + df["long_ret"]

    return df