
# deal_model.py
import pandas as pd
import numpy as np

WEIGHTS = {
    "f_discount_regular": 0.35,
    "f_discount_hist": 0.35,
    "f_value": 0.15,
    "f_rating": 0.10,
    "below_p20": 0.05,
}

def minmax(series: pd.Series) -> pd.Series:
    a = series.astype(float)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return pd.Series([0.5] * len(a), index=a.index)
    return (a - lo) / (hi - lo)

def compute_features(snapshot: pd.DataFrame) -> pd.DataFrame:
    snap = snapshot.copy()
    snap['discount_vs_regular'] = (snap['regular_price'] - snap['price']) / snap['regular_price']
    snap['discount_vs_hist_mean'] = (snap['hist_mean_price'] - snap['price']) / snap['hist_mean_price']
    snap['value_per_100g'] = (snap['pack_size_g'] / snap['price']) * 100
    snap['below_p20'] = (snap['price'] <= snap['hist_p20']).astype(int)

    snap['f_discount_regular'] = minmax(snap['discount_vs_regular'])
    snap['f_discount_hist'] = minmax(snap['discount_vs_hist_mean'])
    snap['f_value'] = minmax(snap['value_per_100g'])
    snap['f_rating'] = minmax(snap['rating'])

    w = WEIGHTS
    snap['deal_score'] = (
        w["f_discount_regular"] * snap['f_discount_regular'] +
        w["f_discount_hist"] * snap['f_discount_hist'] +
        w["f_value"] * snap['f_value'] +
        w["f_rating"] * snap['f_rating'] +
        w["below_p20"] * snap['below_p20"]
    )
    return snap

def rank_deals(snapshot_with_hist: pd.DataFrame, must_have_tags=None, max_per_store=3, top_k=10) -> pd.DataFrame:
    snap = compute_features(snapshot_with_hist)
    base = snap.copy()

    if must_have_tags:
        mask = np.ones(len(base), dtype=bool)
        for t in must_have_tags:
            mask &= base['tags'].str.contains(t, case=False, na=False)
        base = base[mask]

    # one row per product per store (already snapshot), keep top by score
    base = base.sort_values(['product_id', 'store', 'deal_score'], ascending=[True, True, False])
    base = base.groupby(['product_id', 'store']).head(1)
    # limit per store
    base = base.sort_values(['store', 'deal_score'], ascending=[True, False]).groupby('store').head(max_per_store)
    # overall top
    return base.sort_values('deal_score', ascending=False).head(top_k)
