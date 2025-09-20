# app.py (standalone)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------------------
# Data generation (demo)
# ---------------------------
def make_demo_history(days: int = 30) -> pd.DataFrame:
    np.random.seed(7)
    products = [
        {"product_id": "P01", "product": "Protein Bar - Chocolate", "brand": "MacroMax", "tags": "high-protein,low-sugar,gluten-free", "pack_size_g": 60, "category": "bars"},
        {"product_id": "P02", "product": "Tortilla Chips - Sea Salt", "brand": "CasaCrisp", "tags": "vegan,gluten-free", "pack_size_g": 200, "category": "chips"},
        {"product_id": "P03", "product": "Trail Mix - Almond & Cranberry", "brand": "NorthTrail", "tags": "vegan,high-fiber", "pack_size_g": 300, "category": "nuts&mixes"},
        {"product_id": "P04", "product": "Greek Yogurt - Vanilla", "brand": "YogoFarm", "tags": "high-protein", "pack_size_g": 500, "category": "dairy"},
        {"product_id": "P05", "product": "Dark Chocolate 85%", "brand": "CocoaPeak", "tags": "vegan,low-sugar", "pack_size_g": 100, "category": "chocolate"},
    ]
    stores = ["Loblaws", "Walmart", "Costco"]
    start_date = datetime.today() - timedelta(days=days-1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    rows = []
    for p in products:
        base = np.random.uniform(2.0, 12.0)
        for s in stores:
            regular = round((base * (p["pack_size_g"]/100)) * np.random.uniform(0.9, 1.2) + np.random.uniform(0.2, 1.0), 2)
            for d in dates:
                promo = np.random.rand() < 0.15
                noise = np.random.normal(0, 0.15)
                day_price = max(0.5, round(regular * (0.8 if promo else 1.0) + noise, 2))
                rating = round(np.random.uniform(3.6, 4.8), 1)
                sugar = np.random.randint(1, 15)
                protein = np.random.randint(2, 25)
                rows.append({
                    "date": d.date().isoformat(),
                    "store": s,
                    "product_id": p["product_id"],
                    "product": p["product"],
                    "brand": p["brand"],
                    "category": p["category"],
                    "tags": p["tags"],
                    "pack_size_g": p["pack_size_g"],
                    "regular_price": regular,
                    "price": day_price,
                    "rating": rating,
                    "sugar_g": sugar,
                    "protein_g": protein
                })
    return pd.DataFrame(rows)

def make_today_snapshot(history_df: pd.DataFrame) -> pd.DataFrame:
    today = history_df['date'].max()
    snap = history_df[history_df['date'] == today].copy()
    hist = history_df.groupby(['product_id', 'store']).agg(
        hist_mean_price=('price', 'mean'),
        hist_p20=('price', lambda x: float(np.percentile(x, 20))),
        hist_p50=('price', 'median'),
        hist_min=('price', 'min')
    ).reset_index()
    return snap.merge(hist, on=['product_id', 'store'], how='left')

# ---------------------------
# Model / scoring
# ---------------------------
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
        w["below_p20"] * snap['below_p20']
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

    base = base.sort_values(['product_id', 'store', 'deal_score'], ascending=[True, True, False])
    base = base.groupby(['product_id', 'store']).head(1)
    base = base.sort_values(['store', 'deal_score'], ascending=[True, False]).groupby('store').head(max_per_store)
    return base.sort_values('deal_score', ascending=False).head(top_k)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Snack Deal Finder — Demo", layout="wide")
st.title("🍫 Snack Deal Finder — Demo")
st.caption("Ranks snack deals using historical prices, discounts, and a simple AI scoring heuristic.")

@st.cache_data
def load_data():
    hist = make_demo_history(30)
    snap = make_today_snapshot(hist)
    return hist, snap

hist, snap = load_data()

left, right = st.columns([2,1])

with right:
    st.subheader("Filters")
    all_tags = sorted(set(t for tags in snap['tags'].dropna() for t in tags.split(',')))
    diet_tags = st.multiselect("Must-have tags", options=all_tags)

    store_limit = st.slider("Max deals per store", 1, 10, 3)
    top_k = st.slider("Top K overall", 5, 25, 10)

    st.markdown("---")
    st.subheader("About the Score")
    st.write("""
    The deal score blends:
    - Discount vs regular price
    - Discount vs historical mean
    - Value per 100g
    - Product rating
    - Bonus if today's price is below 20th percentile historically
    """)

with left:
    st.subheader("Today’s Best Deals")
    must = diet_tags if diet_tags else None
    deals = rank_deals(snap, must_have_tags=must, max_per_store=store_limit, top_k=top_k)
    st.dataframe(deals[['product','store','price','regular_price','deal_score','tags','rating','pack_size_g']])

    st.markdown("### Price History")
    if len(deals) > 0:
        idx = st.selectbox("Select a deal to view price history", deals.index, format_func=lambda i: f"{deals.loc[i,'product']} @ {deals.loc[i,'store']}")
        sel = deals.loc[idx]
        hist_one = hist[(hist['product_id'] == sel['product_id']) & (hist['store'] == sel['store'])].copy()
        hist_one['date'] = pd.to_datetime(hist_one['date'])

        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(hist_one['date'], hist_one['price'])
        ax.set_title(f"Price history: {sel['product']} @ {sel['store']}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        # ↓ Reduce x-axis tick label size so they fit better
        ax.tick_params(axis='x', labelsize=8)
        st.pyplot(fig)

st.download_button("⬇️ Download Ranked Deals (CSV)", data=deals.to_csv(index=False), file_name="top_snack_deals.csv", mime="text/csv")
