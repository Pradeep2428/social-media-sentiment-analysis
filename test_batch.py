"""Run: python test_batch.py   (needs the model files next to it)"""
import os, re, joblib, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import batch

def clean_text(t):  # same cleaning as app.py
    t = re.sub(r"http\S+|www\S+", "", str(t).lower()); t = re.sub(r"@\w+|<.*?>", "", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t)).strip()

here = os.path.dirname(os.path.abspath(__file__))
m = joblib.load(os.path.join(here, "sentiment_model.pkl")); v = joblib.load(os.path.join(here, "tfidf_vectorizer.pkl"))
df = pd.read_csv(os.path.join(here, "docs", "sample_reviews.csv"))
col = batch.guess_text_column(df); assert col == "review", col
res = batch.analyze_posts(m, v, clean_text, df[col].tolist() + [None, "  "])
assert len(res) == len(df), "empty posts must be skipped"
assert set(res["SentiLens"]) <= {"Positive", "Neutral", "Negative", "Mixed"}
assert res["Changed"].sum() > 0, "context layer should change at least one verdict"
asp = batch.aspect_table(res); assert {"Aspect", "Net score"} <= set(asp.columns)
assert "Aspects" in batch.export_frame(res).columns
print(res[["Model alone", "SentiLens"]].value_counts().head(8)); print(asp.head(8)); print("changed:", res["Changed"].sum(), "of", len(res)); print("OK")
