
# Snack Deal Finder — Streamlit Demo

An end-to-end, deployable **Streamlit** app that ranks snack deals using a simple AI scoring heuristic based on historical prices and product attributes.

## Quickstart (Local)

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

Open the provided URL (usually http://localhost:8501).

## How it works
- `data_gen.py` generates a small synthetic **price history** and a **today snapshot** with historical statistics merged.
- `deal_model.py` computes features (discount vs. regular, discount vs historical mean, value per 100g, rating) and a **deal score**, then ranks.
- `app.py` is a Streamlit UI with:
  - Filters for tags (e.g., vegan, gluten-free)
  - Controls for max deals per store and top-k overall
  - A price-history chart for the selected deal
  - CSV download for the ranked results

## Deploy to Streamlit Community Cloud
1. Push these files to a public GitHub repo.
2. Go to **share.streamlit.io** and choose your repo, set:
   - **Main file:** `app.py`
   - **Python version:** 3.10+
3. The platform will install from `requirements.txt` and launch automatically.

## Replace with real data (optional)
- Swap `make_demo_history()` in `data_gen.py` with your data ingestion (APIs or flyers).
- Keep the snapshot schema: one row per (product_id, store) for today, plus columns: `regular_price, price, rating, pack_size_g, tags` and historical stats (`hist_mean_price, hist_p20, hist_min`).

## Notes
- This is a **demo**; weights in `deal_model.WEIGHTS` are heuristic and can be tuned.
- Add authentication or rate limits if you scrape real sites.
