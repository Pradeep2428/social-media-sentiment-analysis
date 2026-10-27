"""Batch analysis for SentiLens: run the full pipeline over many unlabelled posts.

Pure pandas, no Streamlit, so it can be unit-tested (see test_batch.py).
"""
import pandas as pd
import smart_layer as sl

TEXT_NAMES = ("text", "review", "comment", "post", "tweet", "content", "message")


def guess_text_column(df):
    """Pick the column that holds the posts: a well-known name, else the longest-text column."""
    for c in df.columns:
        if str(c).strip().lower() in TEXT_NAMES:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        return None
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


def analyze_posts(model, vectorizer, clean_fn, texts):
    """One row per non-empty post: base model verdict, SentiLens verdict, why, per-aspect labels."""
    rows = []
    for t in texts:
        t = "" if pd.isna(t) else str(t).strip()
        if not t:
            continue
        r = sl.smart_analyze(model, vectorizer, clean_fn, t, with_explanation=False)
        rows.append({
            "Post": t,
            "Model alone": r["base_label"],
            "SentiLens": r["final"],
            "Changed": r["final"] != r["base_label"],
            "Why": " ".join(r["reasons"]),
            "Aspects": [(a["Aspect"], a["Sentiment"]) for a in r["aspects"]],
        })
    return pd.DataFrame(rows, columns=["Post", "Model alone", "SentiLens", "Changed", "Why", "Aspects"])


def aspect_table(results):
    """Sentiment per named aspect (battery, camera, price...) across all posts.

    Unnamed clause slots ("part 1") are dropped: they carry no topic.
    """
    recs = [(a, s) for lst in results["Aspects"] for a, s in lst if not str(a).startswith("part ")]
    if not recs:
        return pd.DataFrame(columns=["Aspect", "Mentions", "Positive", "Neutral", "Negative", "Net score"])
    df = pd.DataFrame(recs, columns=["Aspect", "Sentiment"])
    ct = pd.crosstab(df["Aspect"], df["Sentiment"]).reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    ct["Mentions"] = ct.sum(axis=1)
    ct["Net score"] = ((ct["Positive"] - ct["Negative"]) / ct["Mentions"]).round(2)
    return ct.reset_index().sort_values("Mentions", ascending=False)[
        ["Aspect", "Mentions", "Positive", "Neutral", "Negative", "Net score"]].reset_index(drop=True)


def export_frame(results):
    """CSV-friendly copy (the aspect list becomes text)."""
    out = results.drop(columns=["Aspects"]).copy()
    out["Aspects"] = results["Aspects"].map(lambda l: "; ".join(f"{a}={s}" for a, s in l))
    return out
