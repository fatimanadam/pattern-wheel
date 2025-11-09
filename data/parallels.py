import numpy as np, pandas as pd

def add_parallel_sims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["trend_id","timestamp"])
    # naive baseline: correlation with same month ±120/±240 across features
    feats = ["gt_search","tiktok_views","youth_proxy","novelty_kw_density","order_kw_density"]
    df["par_sim_10y"] = 0.0
    df["par_sim_20y"] = 0.0
    for tid, g in df.groupby("trend_id"):
        g = g.reset_index(drop=True)
        for i in range(len(g)):
            # look back 120 and 240 months if present
            for delta, col in [(120,"par_sim_10y"), (240,"par_sim_20y")]:
                j = i - delta
                if 0 <= j < len(g):
                    a = g.loc[i,feats].values.astype(float)
                    b = g.loc[j,feats].values.astype(float)
                    num = (a*b).sum()
                    den = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)
                    g.loc[i,col] = num/den
        df.loc[g.index, ["par_sim_10y","par_sim_20y"]] = g[["par_sim_10y","par_sim_20y"]].values
    return df
