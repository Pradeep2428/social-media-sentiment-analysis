"""Temporary experiment runner (opened with ?lab=1). Results are written to lab_results.json."""
import json, os, re, time
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


def run(model, vectorizer, clean_text, train_path, out_path):
    st.title("Lab")
    st.write("classes:", list(model.classes_))
    st.write("model params:", model.get_params())
    st.write("vectorizer params:", {k: str(v) for k, v in vectorizer.get_params().items()})

    cols = ["id", "entity", "sentiment", "text"]
    df = pd.read_csv(train_path, names=cols, header=None).dropna(subset=["text"])
    df = df[df["sentiment"].isin([str(c) for c in model.classes_])].copy()
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["id"]))
    train_all, test = df.iloc[tr_idx], df.iloc[te_idx]
    test = test[~test["clean"].isin(set(train_all["clean"]))]

    # inner validation split (for choosing settings; the test set is only used at the end)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    a, b = next(gss2.split(train_all, groups=train_all["id"]))
    tr, va = train_all.iloc[a], train_all.iloc[b]
    va = va[~va["clean"].isin(set(tr["clean"]))]

    def dedupe(d):
        return d.drop_duplicates(subset=["clean", "sentiment"])

    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import ComplementNB

    def word(ng=(1, 2), mf=None):
        return TfidfVectorizer(ngram_range=ng, sublinear_tf=True, min_df=2, max_df=0.9, max_features=mf)

    def union():
        return FeatureUnion([
            ("word", word()),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=3, max_features=150000)),
        ])

    def lr(c):
        return LogisticRegression(C=c, max_iter=3000, class_weight="balanced")

    configs = {
        "A current settings": lambda: (clone(vectorizer), clone(model), False),
        "D word 1-2, balanced LR C=0.5": lambda: (word(), lr(0.5), True),
        "G word 1-2, balanced LR C=0.25": lambda: (word(), lr(0.25), True),
        "H word 1-3, balanced LR C=0.5": lambda: (word((1, 3)), lr(0.5), True),
        "I word+char, balanced LR C=0.5": lambda: (union(), lr(0.5), True),
        "J word+char, balanced LR C=1": lambda: (union(), lr(1.0), True),
        "K word 1-2, LinearSVC C=0.1": lambda: (word(), LinearSVC(C=0.1, class_weight="balanced"), True),
        "L word 1-2, ComplementNB": lambda: (word(), ComplementNB(alpha=0.3), True),
    }

    results = []
    prog = st.progress(0.0)
    for i, (name, make) in enumerate(configs.items()):
        t0 = time.time()
        vec, clf, dd = make()
        d = dedupe(tr) if dd else tr
        clf.fit(vec.fit_transform(d["clean"]), d["sentiment"])
        p = clf.predict(vec.transform(va["clean"]))
        results.append({
            "config": name, "val_acc": accuracy_score(va["sentiment"], p),
            "val_f1": f1_score(va["sentiment"], p, average="macro"),
            "train_rows": len(d), "features": int(vec.transform(va["clean"][:1]).shape[1]), "secs": round(time.time() - t0, 1),
        })
        prog.progress((i + 1) / len(configs))
        st.write(results[-1])

    best = max(results, key=lambda r: r["val_f1"])
    out = {"validation": results, "best": best["config"], "val_rows": len(va), "test_rows": len(test)}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    st.success(f"best: {best['config']}")
    st.dataframe(pd.DataFrame(results))


def finalize(model, vectorizer, clean_text, train_path, base_dir):
    """Score old vs new settings once on the untouched test set, then save the new model."""
    import joblib
    st.title("Finalize")
    cols = ["id", "entity", "sentiment", "text"]
    df = pd.read_csv(train_path, names=cols, header=None).dropna(subset=["text"])
    df = df[df["sentiment"].isin(["Negative", "Neutral", "Positive"])].copy()
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["id"]))
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    test = test[~test["clean"].isin(set(train["clean"]))]
    labels = ["Negative", "Neutral", "Positive"]

    from sklearn.metrics import confusion_matrix, classification_report

    def new_pipeline():
        return (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9),
                LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=42))

    orig_model = joblib.load(os.path.join(base_dir, "sentiment_model_original.pkl"))
    orig_vec = joblib.load(os.path.join(base_dir, "tfidf_vectorizer_original.pkl"))

    report = {"test_rows": int(len(test)), "labels": labels}
    for name, (vec, clf, dd) in {
        "original": (clone(orig_vec), clone(orig_model), False),
        "improved": (*new_pipeline(), True),
    }.items():
        d = train.drop_duplicates(subset=["clean", "sentiment"]) if dd else train
        clf.fit(vec.fit_transform(d["clean"]), d["sentiment"])
        p = clf.predict(vec.transform(test["clean"]))
        cr = classification_report(test["sentiment"], p, labels=labels, output_dict=True)
        report[name] = {
            "accuracy": float(accuracy_score(test["sentiment"], p)),
            "macro_f1": float(f1_score(test["sentiment"], p, average="macro")),
            "per_class_f1": {l: float(cr[l]["f1-score"]) for l in labels},
            "confusion": confusion_matrix(test["sentiment"], p, labels=labels).tolist(),
            "train_rows": int(len(d)),
        }
        st.write(name, report[name])

    # Deployment model: same settings, refit on all deduplicated data
    vec, clf = new_pipeline()
    alld = df.drop_duplicates(subset=["clean", "sentiment"])
    clf.fit(vec.fit_transform(alld["clean"]), alld["sentiment"])
    joblib.dump(clf, os.path.join(base_dir, "sentiment_model.pkl"))
    joblib.dump(vec, os.path.join(base_dir, "tfidf_vectorizer.pkl"))
    report["deployed_rows"] = int(len(alld))
    report["settings"] = "TF-IDF word 1-2 grams (sublinear tf, min_df=2, max_df=0.9) + Logistic Regression (C=0.5, balanced classes); duplicates removed"
    with open(os.path.join(base_dir, "model_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    st.cache_resource.clear()
    st.success("Saved improved model and model_report.json")


# ---------------------------------------------------------------------------
# Real-tweet check: does the context layer help or hurt on real posts?
# ---------------------------------------------------------------------------

def _splits(clean_text, train_path):
    cols = ["id", "entity", "sentiment", "text"]
    df = pd.read_csv(train_path, names=cols, header=None).dropna(subset=["text"])
    df = df[df["sentiment"].isin(["Negative", "Neutral", "Positive"])].copy()
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(df, groups=df["id"]))
    train_all, test = df.iloc[tr_idx], df.iloc[te_idx]
    test = test[~test["clean"].isin(set(train_all["clean"]))]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    a, b = next(gss2.split(train_all, groups=train_all["id"]))
    tr, va = train_all.iloc[a], train_all.iloc[b]
    va = va[~va["clean"].isin(set(tr["clean"]))]
    return train_all, test, tr, va


def _pipeline():
    return (TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9),
            LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", random_state=42))


RULE_TAGS = [("Sarcasm", "sarcasm"), ("Implicit complaint", "complaint"), ("Negated criticism", "negated criticism"),
             ("Negated praise", "negated praise"), ("Contrasting opinions", "mixed"), ("Hedging", "hedging"),
             ("Clear opinion words", "lexicon override"), ("No opinion words", "neutral fallback")]


def _rule_of(r):
    for key, tag in RULE_TAGS:
        if any(x.startswith(key) for x in r["reasons"]):
            return tag
    return "input change (brand masking / translation)"


def score_layer(sl, clf, vec, clean_text, texts, labels, examples=False):
    """Accuracy of model alone vs SentiLens. Mixed counts as its leaning label."""
    from collections import defaultdict
    base_ok = smart_ok = changed = 0
    fixed = broke = 0
    by_rule = defaultdict(lambda: {"changed": 0, "fixed": 0, "broke": 0, "examples": []})
    for text, y in zip(texts, labels):
        r = sl.smart_analyze(clf, vec, clean_text, text, with_explanation=False)
        s = r["lean"] if r["final"] == "Mixed" else r["final"]
        b_ok, s_ok = r["base_label"] == y, s == y
        base_ok += b_ok; smart_ok += s_ok
        if s != r["base_label"]:
            changed += 1; fixed += (s_ok and not b_ok); broke += (b_ok and not s_ok)
            g = by_rule[_rule_of(r)]
            g["changed"] += 1; g["fixed"] += (s_ok and not b_ok); g["broke"] += (b_ok and not s_ok)
            if examples and b_ok and not s_ok and len(g["examples"]) < 8:
                g["examples"].append(f"[{y} -> {s}] {text[:160]}")
    n = len(labels)
    return {"n": n, "model_acc": base_ok / n, "sentilens_acc": smart_ok / n,
            "changed": changed, "fixed": fixed, "broke": broke,
            "by_rule": {k: dict(v) if examples else {kk: vv for kk, vv in v.items() if kk != "examples"}
                        for k, v in by_rule.items()}}


def real_check(sl, clean_text, train_path, out_path, split="val"):
    st.title(f"Real-tweet check ({split})")
    train_all, test, tr, va = _splits(clean_text, train_path)
    fit_on, eval_on = (tr, va) if split == "val" else (train_all, test)
    fit_on = fit_on.drop_duplicates(subset=["clean", "sentiment"])
    vec, clf = _pipeline()
    clf.fit(vec.fit_transform(fit_on["clean"]), fit_on["sentiment"])
    res = score_layer(sl, clf, vec, clean_text, eval_on["text"].astype(str).tolist(), eval_on["sentiment"].tolist(),
                      examples=(split == "val"))
    res["split"] = split
    st.write(res)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)


# ---------------------------------------------------------------------------
# Hinglish: SemEval-2020 Task 9 (SentiMix) + Hinglish YouTube comments
# ---------------------------------------------------------------------------

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
SENTIMIX_FILES = ["train_14k_split_conll.txt", "dev_3k_split_conll.txt",
                  "Hindi_test_unalbelled_conll_updated.txt", "test_labels_hinglish.txt"]


def download_hinglish(data_dir):
    import urllib.request
    os.makedirs(data_dir, exist_ok=True)
    files = [("RTT1/SentiMix", f) for f in SENTIMIX_FILES] + \
            [("shae2977/hinglish-youtube-sentiments-dataset", "yt_hinglish_comments_dataset.csv")]
    for repo, name in files:
        dest = os.path.join(data_dir, name)
        if not os.path.exists(dest):
            urllib.request.urlretrieve(HF.format(repo=repo, file=name), dest)
    return {name: os.path.getsize(os.path.join(data_dir, name)) for _, name in files}


def read_conll(path):
    """SentiMix CoNLL: 'meta <uid> [label]' line, then one 'token<TAB>lang' line per token."""
    rows, cur = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("meta"):
                if cur: rows.append(cur)
                parts = line.split("\t")
                cur = {"uid": parts[1].strip(), "label": parts[2].strip().title() if len(parts) > 2 and parts[2].strip() else None,
                       "tokens": [], "langs": []}
            elif line.strip() and cur is not None:
                parts = line.split("\t")
                cur["tokens"].append(parts[0])
                cur["langs"].append(parts[1] if len(parts) > 1 else "")
    if cur: rows.append(cur)
    return rows


def _detok(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"@ (\w)", r"@\1", text)
    text = re.sub(r"# (\w)", r"#\1", text)
    return text


def _load_sentimix(data_dir):
    tr = [r for r in read_conll(os.path.join(data_dir, "train_14k_split_conll.txt")) if r["label"]]
    dv = [r for r in read_conll(os.path.join(data_dir, "dev_3k_split_conll.txt")) if r["label"]]
    te = read_conll(os.path.join(data_dir, "Hindi_test_unalbelled_conll_updated.txt"))
    lab = pd.read_csv(os.path.join(data_dir, "test_labels_hinglish.txt"))
    lab.columns = [c.strip().lower() for c in lab.columns]
    lmap = dict(zip(lab.iloc[:, 0].astype(str).str.strip(), lab.iloc[:, 1].astype(str).str.strip().str.title()))
    for r in te:
        r["label"] = lmap.get(r["uid"])
    te = [r for r in te if r["label"]]
    return tr, dv, te


def learn_hinglish_lexicon(train_rows, min_count=10, min_logodds=1.0, exclude=(), english_vocab=()):
    """Hindi-tagged words that lean strongly positive or negative in the training tweets."""
    import math
    from collections import Counter
    pos, neg, total, hin = Counter(), Counter(), Counter(), Counter()
    for r in train_rows:
        seen = set()
        for tok, lang in zip(r["tokens"], r["langs"]):
            w = tok.lower()
            if not re.fullmatch(r"[a-z]{2,}", w) or w in seen:
                continue
            seen.add(w)
            total[w] += 1
            hin[w] += lang.lower().startswith("hin")
            if r["label"] == "Positive": pos[w] += 1
            elif r["label"] == "Negative": neg[w] += 1
    n_pos = sum(1 for r in train_rows if r["label"] == "Positive")
    n_neg = sum(1 for r in train_rows if r["label"] == "Negative")
    lex = {}
    for w, c in total.items():
        if c < min_count or hin[w] / c < 0.8 or w in exclude or w in english_vocab:
            continue
        lo = math.log(((pos[w] + 1) / (n_pos + 2)) / ((neg[w] + 1) / (n_neg + 2)))
        if abs(lo) >= min_logodds:
            lex[w] = round(lo, 3)
    return lex


def _eval_rows(sl, model, vectorizer, clean_text, texts, labels):
    from sklearn.metrics import f1_score
    base, smart = [], []
    for t in texts:
        r = sl.smart_analyze(model, vectorizer, clean_text, t, with_explanation=False)
        base.append(r["base_label"])
        smart.append(r["lean"] if r["final"] == "Mixed" else r["final"])
    acc = lambda p: sum(a == b for a, b in zip(p, labels)) / len(labels)
    f1 = lambda p: f1_score(labels, p, average="macro", labels=["Negative", "Neutral", "Positive"])
    return {"model_acc": acc(base), "model_f1": f1(base), "sentilens_acc": acc(smart), "sentilens_f1": f1(smart)}


def hinglish_phase(sl, model, vectorizer, clean_text, base_dir, phase="dev"):
    st.title(f"Hinglish ({phase})")
    data_dir = os.path.join(base_dir, "..", "dataset", "hinglish")
    st.write("files", download_hinglish(data_dir))
    tr, dv, te = _load_sentimix(data_dir)
    st.write({"train": len(tr), "dev": len(dv), "test": len(te),
              "labels": pd.Series([r["label"] for r in tr]).value_counts().to_dict()})
    yt = pd.read_csv(os.path.join(data_dir, "yt_hinglish_comments_dataset.csv"))
    st.write("youtube columns", list(yt.columns), yt.head(3))
    excl_path = os.path.join(base_dir, "hinglish_exclude.json")
    exclude = set(json.load(open(excl_path))) if os.path.exists(excl_path) else set()
    lex_path = os.path.join(base_dir, "hinglish_lexicon.json")
    english = {w for w in vectorizer.vocabulary_ if " " not in w}  # words the English model already knows

    if phase == "dev":
        dv_texts, dv_labels = [_detok(r["tokens"]) for r in dv], [r["label"] for r in dv]
        results = []
        sl.set_learned_hinglish({})
        results.append({"config": "built-in dictionary only", **_eval_rows(sl, model, vectorizer, clean_text, dv_texts, dv_labels)})
        best = None
        for mc in (10, 25):
            for lo in (0.8, 1.2):
                lex = learn_hinglish_lexicon(tr, mc, lo, exclude, english)
                sl.set_learned_hinglish(lex)
                r = {"config": f"learned lexicon min_count={mc} log-odds>={lo} ({len(lex)} words)",
                     "min_count": mc, "min_logodds": lo,
                     **_eval_rows(sl, model, vectorizer, clean_text, dv_texts, dv_labels)}
                results.append(r)
                if best is None or r["sentilens_f1"] > best["sentilens_f1"]:
                    best = r
                st.write(r)
        lex = learn_hinglish_lexicon(tr, best["min_count"], best["min_logodds"], exclude, english)
        top = sorted(lex.items(), key=lambda kv: kv[1])
        st.write("most negative", top[:60])
        st.write("most positive", top[-60:][::-1])
        json.dump({"min_count": best["min_count"], "min_logodds": best["min_logodds"], "lexicon": lex},
                  open(lex_path, "w"), indent=0)
        json.dump({"dev": results}, open(os.path.join(base_dir, "hinglish_dev.json"), "w"), indent=1)
        sl.set_learned_hinglish({})
        st.success(f"chosen: {best['config']}")
    else:
        saved = json.load(open(lex_path))
        lex = learn_hinglish_lexicon(tr, saved["min_count"], saved["min_logodds"], exclude, english)
        json.dump({**saved, "lexicon": lex}, open(lex_path, "w"), indent=0)
        te_texts, te_labels = [_detok(r["tokens"]) for r in te], [r["label"] for r in te]
        out = {"test_n": len(te), "lexicon_words": len(lex)}
        sl.set_learned_hinglish({})
        out["semeval_builtin"] = _eval_rows(sl, model, vectorizer, clean_text, te_texts, te_labels)
        sl.set_learned_hinglish(lex)
        out["semeval_learned"] = _eval_rows(sl, model, vectorizer, clean_text, te_texts, te_labels)
        # YouTube comments: outside check, never used for tuning
        tcol, lcol = "comment", "sentiment"
        lab = yt[lcol].astype(str).str.strip().str.title().replace({"0": "Negative", "1": "Neutral", "2": "Positive"})
        keep = lab.isin(["Negative", "Neutral", "Positive"])
        out["youtube_n"] = int(keep.sum())
        out["youtube_columns"] = [tcol, lcol]
        sl.set_learned_hinglish({})
        out["youtube_builtin"] = _eval_rows(sl, model, vectorizer, clean_text, yt[tcol][keep].astype(str).tolist(), lab[keep].tolist())
        sl.set_learned_hinglish(lex)
        out["youtube_learned"] = _eval_rows(sl, model, vectorizer, clean_text, yt[tcol][keep].astype(str).tolist(), lab[keep].tolist())
        # Reference point: a TF-IDF model trained directly on Hinglish tweets
        vec, clf = _pipeline()
        tr_texts = [_detok(r["tokens"]) for r in tr]
        clf.fit(vec.fit_transform([clean_text(t) for t in tr_texts]), [r["label"] for r in tr])
        p = clf.predict(vec.transform([clean_text(t) for t in te_texts]))
        from sklearn.metrics import f1_score
        out["reference_trained_on_hinglish"] = {
            "acc": float(sum(a == b for a, b in zip(p, te_labels)) / len(te_labels)),
            "f1": float(f1_score(te_labels, p, average="macro"))}
        json.dump(out, open(os.path.join(base_dir, "hinglish_eval.json"), "w"), indent=1)
        st.write(out)
