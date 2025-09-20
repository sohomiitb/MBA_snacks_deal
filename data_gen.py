
# data_gen.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
        base = np.random.uniform(2.0, 12.0)  # base per 100g price proxy
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

if __name__ == "__main__":
    df = make_demo_history(30)
    snap = make_today_snapshot(df)
    df.to_csv("snack_prices_demo.csv", index=False)
    snap.to_csv("snack_today_snapshot.csv", index=False)
    print("Wrote snack_prices_demo.csv and snack_today_snapshot.csv")
