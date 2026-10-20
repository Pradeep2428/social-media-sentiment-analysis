"""
Smart context layer for the TF-IDF + Logistic Regression sentiment model.

The base model only sees a bag of words, so it struggles with:
  * sarcasm            ("Great job! Another two-hour delay.")
  * hedged complaints  ("It works, I guess, but I wouldn't recommend it.")
  * mixed opinions     ("Excellent camera, but disappointing battery.")
  * Hinglish / emoji   ("ekdum mast hai 🔥")

This module adds a transparent, rule-based layer on top of the model:
  1. normalize_text()  - translates Hinglish words and emojis into English cues
  2. explain()         - word-level contributions from the model's own weights
  3. detect_*()        - context checks for sarcasm, hedging and contrast
  4. smart_analyze()   - combines everything into a final, explained verdict
"""

import re
import numpy as np

# ---------------------------------------------------------------------------
# 1. Normalisation: Hinglish + emoji -> English cues the model understands
# ---------------------------------------------------------------------------

HINGLISH = {
    # positive
    "mast": "great", "badhiya": "great", "badiya": "great", "zabardast": "awesome",
    "jhakaas": "awesome", "kamaal": "amazing", "kamal": "amazing", "shandar": "excellent",
    "shaandar": "excellent", "accha": "good", "achha": "good", "acha": "good",
    "achha": "good", "sundar": "beautiful", "pyaar": "love", "pyar": "love",
    "khush": "happy", "maza": "fun", "mazaa": "fun", "behtareen": "excellent",
    "lajawab": "excellent", "sahi": "good",
    # negative
    "bekar": "bad", "bekaar": "bad", "bakwas": "terrible", "bakwaas": "terrible",
    "ganda": "bad", "gandi": "bad", "faltu": "useless", "faaltu": "useless",
    "ghatiya": "worst", "bura": "bad", "buri": "bad", "kharab": "broken",
    "kharaab": "broken", "dukhi": "sad", "gussa": "angry", "barbad": "waste",
    "barbaad": "waste", "dhoka": "fraud", "dhokha": "fraud", "bekaar": "bad",
    "pareshan": "annoyed", "ghatia": "worst", "chutiyapa": "nonsense",
    # modifiers
    "nahi": "not", "nahin": "not", "nai": "not", "mat": "dont",
    "bahut": "very", "bohot": "very", "bahot": "very", "bht": "very",
    "ekdum": "totally", "bilkul": "totally", "thoda": "slightly",
}

HINGLISH_PHRASES = {
    "paisa vasool": "worth it",
    "paisa wasool": "worth it",
    "time waste": "waste of time",
    "paisa barbad": "waste of money",
    "paisa barbaad": "waste of money",
}

# Hindi filler words that carry no sentiment - dropped so they don't add noise
HINDI_FILLERS = {"hai", "tha", "thi", "the", "ye", "yeh", "wo", "woh", "bhi",
                 "kya", "se", "ka", "ki", "ke", "ko", "aur", "par", "hi", "toh", "to"}
# "the" and "to" are English too - keep them
HINDI_FILLERS -= {"the", "to"}

EMOJI = {
    "🔥": "amazing", "😍": "love", "🥰": "love", "❤️": "love", "❤": "love",
    "💯": "perfect", "👏": "great", "👍": "good", "😊": "happy", "🙂": "happy",
    "😀": "happy", "😁": "happy", "😄": "happy", "🤩": "amazing", "🥳": "happy",
    "😂": "funny", "🤣": "funny", "✨": "nice",
    "😡": "angry", "😠": "angry", "🤬": "hate", "👎": "bad", "😢": "sad",
    "😭": "sad", "💔": "heartbroken", "🤮": "disgusting", "😤": "annoyed",
    "😞": "disappointed", "😩": "tired", "😫": "tired",
    "😐": "meh", "😑": "meh",
}
SARCASM_EMOJI = {"🙄", "😒", "🙃", "😏"}


# Hinglish opinion words learned from the SemEval-2020 SentiMix training tweets
# (see lab_experiments.learn_hinglish_lexicon). Positive weight -> "good", negative -> "bad".
LEARNED_HINGLISH = {}


def set_learned_hinglish(lexicon):
    LEARNED_HINGLISH.clear()
    LEARNED_HINGLISH.update(lexicon or {})


def _load_learned_hinglish():
    import json, os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hinglish_lexicon.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                set_learned_hinglish(json.load(f).get("lexicon", {}))
        except (OSError, ValueError):
            pass


def normalize_text(text):
    """Return (normalized_text, list_of_changes) for display."""
    changes = []
    out = str(text)

    for emo, word in EMOJI.items():
        if emo in out:
            out = out.replace(emo, f" {word} ")
            changes.append((emo, word))
    for emo in SARCASM_EMOJI:
        if emo in out:
            out = out.replace(emo, " ")
            changes.append((emo, "sarcasm cue"))

    low = out.lower()
    for phrase, eng in HINGLISH_PHRASES.items():
        if phrase in low:
            low = low.replace(phrase, eng)
            changes.append((phrase, eng))

    tokens = re.findall(r"[\w']+|[^\w\s]", low)
    new_tokens = []
    for tok in tokens:
        if tok in HINGLISH:
            changes.append((tok, HINGLISH[tok]))
            new_tokens.append(HINGLISH[tok])
        elif tok in LEARNED_HINGLISH:
            eng = "good" if LEARNED_HINGLISH[tok] > 0 else "bad"
            changes.append((tok, eng + " (learned)"))
            new_tokens.append(eng)
        elif tok in HINDI_FILLERS:
            continue
        else:
            new_tokens.append(tok)
    normalized = " ".join(new_tokens)
    normalized = re.sub(r"\s+([^\w\s])", r"\1", normalized)
    return normalized, changes


# ---------------------------------------------------------------------------
# 2. Explainability from Logistic Regression weights
# ---------------------------------------------------------------------------

def _class_index(model, name):
    classes = [str(c) for c in model.classes_]
    return classes.index(name) if name in classes else None


_NAMES_CACHE = {}


def _feature_names(vectorizer):
    """get_feature_names_out() rebuilds a large array on every call, so keep one copy per vectorizer."""
    key = id(vectorizer)
    if key not in _NAMES_CACHE:
        _NAMES_CACHE[key] = vectorizer.get_feature_names_out()
    return _NAMES_CACHE[key]


def explain(model, vectorizer, cleaned_text, top_k=8):
    """
    Word-level contributions using the model's own coefficients.

    For each TF-IDF feature present in the text we compute
        polarity = tfidf_value * (coef[Positive] - coef[Negative])
    so > 0 pushes towards Positive and < 0 pushes towards Negative.
    Returns (token_scores, top_features) or (None, None) if unsupported.
    """
    if not hasattr(model, "coef_"):
        return None, None
    pos_i, neg_i = _class_index(model, "Positive"), _class_index(model, "Negative")
    if pos_i is None or neg_i is None:
        return None, None

    x = vectorizer.transform([cleaned_text]).tocsr()
    names = _feature_names(vectorizer)
    direction = model.coef_[pos_i] - model.coef_[neg_i]

    features = []
    for idx, val in zip(x.indices, x.data):
        features.append((names[idx], float(val * direction[idx])))
    features.sort(key=lambda f: abs(f[1]), reverse=True)

    # Spread bigram scores onto their words for the highlighted sentence
    token_scores = {}
    for term, score in features:
        parts = term.split()
        share = score / len(parts)
        for p in parts:
            token_scores[p] = token_scores.get(p, 0.0) + share
    return token_scores, features[:top_k]


def highlight_html(cleaned_text, token_scores):
    """Colour each word green (pushes Positive) or red (pushes Negative)."""
    if not token_scores:
        return cleaned_text
    max_abs = max((abs(v) for v in token_scores.values()), default=1.0) or 1.0
    spans = []
    for word in cleaned_text.split():
        s = token_scores.get(word, 0.0)
        alpha = min(abs(s) / max_abs, 1.0) * 0.75 + (0.12 if s else 0)
        if s > 0:
            colour = f"rgba(42,157,143,{alpha:.2f})"
        elif s < 0:
            colour = f"rgba(231,111,81,{alpha:.2f})"
        else:
            colour = "transparent"
        title = f"{s:+.3f}" if s else "no learned weight"
        spans.append(
            f'<span title="{title}" style="background:{colour};padding:2px 5px;'
            f'border-radius:6px;margin:2px;display:inline-block">{word}</span>'
        )
    return " ".join(spans)


# ---------------------------------------------------------------------------
# 3. Context checks
# ---------------------------------------------------------------------------

POSITIVE_CUES = [
    "great", "love", "loving", "awesome", "fantastic", "perfect", "wonderful",
    "amazing", "brilliant", "nice", "thanks", "thank you", "best", "excellent",
    "fabulous", "lovely", "so happy", "just what i needed", "exactly what i wanted",
]
# Real complaint words - a bad thing actually happened
NEGATIVE_SITUATIONS = [
    "delay", "delayed", "late", "broken", "broke", "crash", "crashed", "crashes",
    "stuck", "cancelled", "canceled", "lost", "bug", "bugs", "error", "failed",
    "fails", "outage", "overcharged", "charged twice", "slow", "traffic", "queue",
    "no signal", "battery died", "spam", "rejected", "deleted", "deleting",
    "refund", "damaged", "missed", "stolen", "hacked", "logged out", "ban", "banned",
]
# Words that make a complaint sound repeated or drawn-out (not negative on their own)
AMPLIFIERS = ["again", "another", "hours", "still", "as usual", "every time", "once again"]
SARCASM_MARKERS = [
    "yeah right", "oh great", "oh wonderful", "just what i needed",
    "just what i wanted", "thanks a lot", "thanks for nothing", "love how",
    "love it when", "so glad", "what a surprise", "how wonderful", "/s",
    "said no one", "because that's exactly", "my favourite thing", "my favorite thing",
    "great job", "well done",
]
STRONG_HEDGES = [
    "wouldn't recommend", "would not recommend", "won't recommend", "wont recommend",
    "do not recommend", "don't recommend", "dont recommend", "can't recommend",
    "not worth", "waste of", "never again", "won't buy", "wont buy", "not buying",
    "expected more", "expected better", "not impressed", "save your money",
    "returning it", "returned it", "regret", "overpriced", "avoid",
    "could have been better", "could be better",
]
WEAK_HEDGES = ["i guess", "meh", "it's okay", "its okay", "it's ok", "its ok",
               "not great", "not that good", "kind of", "sort of", "nothing special"]

# Negated criticism that actually means praise ("not bad" = good)
LITOTES = ["not bad", "not too bad", "not bad at all", "not half bad", "not terrible",
           "can't complain", "cant complain", "no complaints", "nothing to complain",
           "not disappointed", "wasn't disappointed", "was not disappointed", "never disappoints"]

CONTRAST_SPLIT = r"\b(?:but|however|although|though|yet|whereas|except|on the other hand)\b"

ASPECTS = [
    "camera", "battery", "price", "service", "delivery", "screen", "display",
    "food", "staff", "quality", "performance", "design", "sound", "audio",
    "app", "update", "support", "story", "acting", "music", "graphics",
    "packaging", "speed", "ambience", "plot", "ending", "gameplay", "size",
]


def _find(text, phrases):
    hits = []
    for p in phrases:
        if re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", text):
            hits.append(p)
    return hits


def _plain(text):
    return str(text).lower().replace("’", "'").replace("‘", "'")


def detect_sarcasm(raw_text, normalized_text):
    t = _plain(normalized_text)
    markers = _find(t, SARCASM_MARKERS)
    pos = _find(t, POSITIVE_CUES)
    neg = _find(t, NEGATIVE_SITUATIONS)
    emoji = [e for e in SARCASM_EMOJI if e in str(raw_text)]
    amp = _find(t, AMPLIFIERS)
    flagged = (
        bool(emoji)
        or bool(markers and neg)
        or bool(pos and neg and (markers or amp or "!" in str(raw_text)))
    )
    evidence = []
    if markers: evidence.append("sarcastic phrase: " + ", ".join(f"“{m}”" for m in markers))
    if pos and neg: evidence.append("praise words (" + ", ".join(pos[:3]) + ") next to a complaint (" + ", ".join(neg[:3]) + ")")
    if emoji: evidence.append("eye-roll style emoji " + " ".join(emoji))
    return flagged, evidence


def detect_hedging(normalized_text):
    t = _plain(normalized_text)
    strong = _find(t, STRONG_HEDGES)
    weak = _find(t, WEAK_HEDGES)
    return strong, weak


def split_clauses(text):
    parts = [p.strip(" ,.;:-!?") for p in re.split(CONTRAST_SPLIT, _plain(text))]
    return [p for p in parts if len(p.split()) >= 2]


def aspect_of(clause, i):
    for a in ASPECTS:
        if re.search(r"\b" + a + r"\b", clause):
            return a
    return f"part {i + 1}"


# ---------------------------------------------------------------------------
# 3b. General opinion lexicon (VADER, Hutto & Gilbert 2014, MIT licence)
# ---------------------------------------------------------------------------

import os as _os

_AMBIGUOUS = {"like", "please", "yes", "ok", "okay", "well", "pretty", "sure", "matter",
              "play", "god", "kind", "joke", "no",
              # discourse markers and intensifiers, not opinions
              "honestly", "really", "truly", "seriously", "actually", "indeed", "certainly",
              "definitely", "totally", "absolutely", "true", "huge", "big", "share"}
# Common opinion words missing from VADER
_EXTRA = {"mediocre": -1.8, "rubbish": -2.0, "meh": -1.0, "lit": 1.5, "fire": 0.0}


def _load_vader():
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vader_lexicon.txt")
    lex = {}
    if _os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t")
                if len(parts) >= 2 and re.fullmatch(r"[a-z][a-z'-]*", parts[0]) and parts[0] not in _AMBIGUOUS:
                    try:
                        lex[parts[0]] = float(parts[1])
                    except ValueError:
                        pass
    lex.update({k: v for k, v in _EXTRA.items() if v})
    lex.pop("fire", None)  # "fire" is slang praise online, but a disaster in news - leave it to the model
    return lex


VADER = _load_vader()
_load_learned_hinglish()
NEGATORS = {"not", "no", "never", "nothing", "neither", "nor", "hardly", "barely", "cannot",
            "cant", "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "wont",
            "wouldnt", "shouldnt", "couldnt", "aint", "without"}
NEGATION_WINDOW = 4  # a negator flips opinion words up to 4 words after it
CLAUSE_BREAK = r"[,.;:!?]|\b(?:but|however|although|though|yet|whereas)\b"


def _tokens(text):
    return re.findall(r"[a-z][a-z'-]*", _plain(text))


def _is_negator(tok, prev):
    return tok in NEGATORS or tok.endswith("n't") or (prev == "far" and tok == "from")


def lexicon_profile(text):
    """
    Score a text with VADER word ratings, flipping words that sit after a negator
    in the same clause. Returns dict with total score and which kinds of words were seen.
    """
    prof = {"score": 0.0, "pos": [], "neg": [], "neg_pos": [], "neg_neg": []}
    for seg in re.split(CLAUSE_BREAK, _plain(text)):
        if not seg or not seg.strip():
            continue
        since_neg, prev = 99, ""
        for tok in _tokens(seg):
            since_neg += 1
            negated = since_neg <= NEGATION_WINDOW
            if _is_negator(tok, prev):
                since_neg = 0
            elif tok in VADER:
                v = VADER[tok]
                if abs(v) >= 1.0:
                    if negated:
                        prof["score"] -= 0.74 * v
                        prof["neg_pos" if v > 0 else "neg_neg"].append(tok)
                    else:
                        prof["score"] += v
                        prof["pos" if v > 0 else "neg"].append(tok)
            prev = tok
    return prof


def lexicon_polarity(clause):
    p = lexicon_profile(clause)["score"]
    return "Positive" if p >= 1.0 else "Negative" if p <= -1.0 else None


# Brands and games from the training data. The model learned opinions about these
# names themselves, so they are hidden before prediction.
TRAINING_ENTITIES = [
    "amazon", "apex legends", "apexlegends", "apex", "assassins creed", "assassin's creed",
    "assassinscreed", "battlefield", "borderlands", "cs-go", "csgo", "cs go", "call of duty",
    "callofduty", "black ops", "cold war", "cyberpunk 2077", "cyberpunk2077", "cyberpunk",
    "dota 2", "dota2", "dota", "fifa", "facebook", "fortnite", "google", "grand theft auto",
    "gta v", "gta", "hearthstone", "home depot", "homedepot", "league of legends", "madden nfl",
    "madden", "microsoft", "nba 2k", "nba2k", "nvidia", "overwatch", "playstation 5",
    "playstation", "ps5", "ps4", "pubg", "red dead redemption", "red dead", "rdr2",
    "ghost recon", "rainbow six", "rainbow 6", "verizon", "world of warcraft", "warcraft",
    "xbox", "johnson & johnson", "johnson&johnson", "johnson and johnson",
]


def mask_entities(text):
    found = []
    out = str(text)
    for name in sorted(TRAINING_ENTITIES, key=len, reverse=True):
        pat = r"(?<![\w])" + re.escape(name) + r"(?![\w])"
        if re.search(pat, out, flags=re.I):
            found.append(name)
            out = re.sub(pat, " ", out, flags=re.I)
    return re.sub(r"\s+", " ", out).strip(), found


SARCASM_OPENERS = r"^(?:oh\s+)?(?:wow|great|fantastic|brilliant|amazing|perfect|wonderful|awesome|lovely|" \
                  r"thanks(?: a lot)?|thank you|just amazing|best day ever|so glad|love it when|love how|" \
                  r"love that|nice one|nice|well done|great job|congrats)\b"
COMPLAINT_EVENT = re.compile(
    r"\b(?:crash\w*|froze|frozen|freez\w*|broke|broken|died|dead|stopped working|stopped \w+ing|"
    r"overheat\w*|lost|cancel+ed|late|delay\w*|fail\w*|logged me out|error\w*|bug\w*|stuck|"
    r"outage|went down|went offline|offline|hang\w*|lag\w*|drain\w*|deleted|wiped|charged|"
    r"refund\w*|missed|leak\w*|spam\w*|hack\w*|buffer\w*|disconnect\w*)\b")

COMPLAINT_PATTERNS = [
    r"\breturn(?:ing|ed)? (?:it|this|the|them)\b", r"\bsend(?:ing)? (?:it|this) back\b",
    r"\bexpected (?:a lot |much |way |so )?(?:more|better)\b", r"\bregret\w*\b",
    r"\bnot (?:\w+ ){0,3}worth\b", r"\b(?:money|cash) back\b", r"\bwaste of\b",
    r"\b(?:can't|cannot|cant|won't|wont|wouldn't|wouldnt|don't|dont|do not|would not) (?:really |honestly )?recommend\b",
    r"\b(?:won't|wont|will not|never) (?:be )?(?:buy|buying|order|ordering|use|using) (?:\w+ )?again\b",
    r"\bwould(?: not|n't|nt) (?:buy|order|use) (?:\w+ ){0,3}again\b", r"\bnever again\b",
    r"\bswitch(?:ing|ed)? (?:away|to another|back)\b", r"\bunsubscrib\w*\b", r"\bnot impressed\b",
    r"\bsave your money\b", r"\boverpriced\b", r"\bavoid\b", r"\bcould (?:have been|be) better\b",
]


def find_complaints(text):
    t = _plain(text)
    hits = [m.group(0) for p in COMPLAINT_PATTERNS for m in [re.search(p, t)] if m]
    return hits + [h for h in STRONG_HEDGES if h in t and h not in " ".join(hits)]


def detect_sarcasm_v2(raw_text, normalized_text, allow_opener=True):
    """Original checks plus: praising opener followed by something going wrong."""
    flagged, evidence = detect_sarcasm(raw_text, normalized_text)
    t = _plain(normalized_text).strip()
    opener = re.match(SARCASM_OPENERS, t)
    has_contrast = re.search(r"\b(?:but|however|although|though|whereas)\b", t)
    if opener and not has_contrast and allow_opener:  # "Great phone but awful battery" is mixed, not sarcastic
        rest = t[opener.end():]
        event = COMPLAINT_EVENT.search(rest)  # something has to go wrong, not just a negative adjective
        if event:
            what = event.group(0)
            if not flagged:
                flagged = True
            evidence.append(f"praising opener “{opener.group(0)}” followed by something going wrong (“{what}”)")
    return flagged, evidence


# ---------------------------------------------------------------------------
# 4. Combined analysis
# ---------------------------------------------------------------------------

def predict(model, vectorizer, clean_fn, text):
    feats = vectorizer.transform([clean_fn(text)])
    probs = model.predict_proba(feats)[0]
    classes = [str(c) for c in model.classes_]
    best = int(np.argmax(probs))
    return classes[best], float(probs[best]), dict(zip(classes, map(float, probs)))


SHORT_POST_WORDS = 15


def smart_analyze(model, vectorizer, clean_fn, raw_text, uncertain_below=0.50, with_explanation=True):
    normalized, changes = normalize_text(raw_text)
    model_text, masked = mask_entities(normalized)
    cleaned = clean_fn(model_text)

    base_label, base_conf, base_probs = predict(model, vectorizer, clean_fn, raw_text)
    norm_label, norm_conf, norm_probs = predict(model, vectorizer, clean_fn, model_text)

    final, reasons = norm_label, []
    if changes:
        reasons.append("Translated Hinglish words / emojis into English cues before prediction.")
    if masked:
        reasons.append("Ignored brand names the model learned biased opinions about (" + ", ".join(masked) + ").")

    prof = lexicon_profile(normalized)
    # Override rules are only trusted on short posts. In long posts the model sees more
    # evidence than a handful of opinion words, and the rules misfire (checked on real tweets).
    n_words = len(_tokens(normalized))
    short, very_short = n_words <= SHORT_POST_WORDS, n_words <= 10
    sarcastic, sarcasm_evidence = detect_sarcasm_v2(raw_text, normalized, allow_opener=short)
    strong_hedge = find_complaints(normalized)
    weak_hedge = detect_hedging(normalized)[1]

    # Clause breakdown: contrast words use model + lexicon; comma splits use the lexicon only
    aspects = []
    clauses = split_clauses(normalized)
    if len(clauses) >= 2:
        for i, c in enumerate(clauses):
            lab, conf, _ = predict(model, vectorizer, clean_fn, mask_entities(c)[0])
            source = "model"
            lex = lexicon_polarity(c)
            if lex and lex != lab:
                lab, source, conf = lex, "opinion words", None
            aspects.append({"Aspect": aspect_of(c, i), "Clause": c, "Sentiment": lab,
                            "Confidence": conf, "Decided by": source})
    else:
        parts = [p.strip() for p in re.split(r"[,;]", _plain(normalized)) if len(p.split()) >= 2]
        pols = [(p, lexicon_polarity(p)) for p in parts]
        if short and len(parts) >= 2 and {"Positive", "Negative"} <= {l for _, l in pols}:
            for i, (p, l) in enumerate(pols):
                if l:
                    aspects.append({"Aspect": aspect_of(p, i), "Clause": p, "Sentiment": l,
                                    "Confidence": None, "Decided by": "opinion words"})
    labels = {a["Sentiment"] for a in aspects} - {"Neutral"}
    is_mixed = len(labels) >= 2
    lean = None
    if is_mixed:
        lean = norm_label if norm_label in labels else aspects[-1]["Sentiment"]

    litotes = _find(_plain(normalized), LITOTES)
    litotes = [h for h in litotes if not any(h != o and h in o for o in litotes)]
    only_negated_neg = prof["neg_neg"] and not prof["neg"]
    only_negated_pos = prof["neg_pos"] and not prof["pos"]

    if sarcastic and final != "Negative":
        final = "Negative"
        reasons.append("Sarcasm detected: positive words are used about a bad situation.")
    elif strong_hedge and final != "Negative":  # a stated complaint outweighs a mixed reading
        final = "Negative"
        reasons.append("Implicit complaint: " + ", ".join(f"“{h}”" for h in strong_hedge[:3]) + " signals dissatisfaction.")
    elif (litotes or (only_negated_neg and short)) and final == "Negative" and not is_mixed:
        final = "Positive" if prof["pos"] else "Neutral"
        what = litotes or [f"not {w}" for w in prof["neg_neg"]]
        reasons.append("Negated criticism: " + ", ".join(f"“{h}”" for h in what[:2]) + " is not a complaint.")
    elif only_negated_pos and short and final == "Positive" and not is_mixed:
        final = "Negative" if norm_probs.get("Negative", 0) >= norm_probs.get("Neutral", 0) else "Neutral"
        reasons.append("Negated praise: " + ", ".join(f"“not {w}”" for w in prof["neg_pos"][:2]) + " is not a compliment.")
    elif is_mixed:
        final = "Mixed"
        reasons.append(f"Contrasting opinions found; overall it leans {lean}.")
    elif weak_hedge and short and final == "Positive":
        final = "Neutral"
        reasons.append("Hedging words (" + ", ".join(weak_hedge) + ") weaken the praise.")
    else:
        lex_label = "Positive" if prof["score"] >= 1.5 else "Negative" if prof["score"] <= -1.5 else None
        has_opinion = (prof["pos"] or prof["neg"] or prof["neg_pos"] or prof["neg_neg"]
                       or strong_hedge or sarcastic or litotes)  # a detected complaint is an opinion too
        if lex_label and lex_label != final and norm_conf < 0.50 and very_short and abs(prof["score"]) >= 2.0:
            final = lex_label
            reasons.append(f"Clear opinion words ({', '.join((prof['pos'] or prof['neg'])[:3])}) outweigh an unsure model ({norm_conf*100:.0f}%).")
        elif not has_opinion and not changes and not masked and final != "Neutral" and norm_conf < 0.45 and very_short:
            final = "Neutral"
            reasons.append(f"No opinion words found and the model is unsure ({norm_conf*100:.0f}%), so this reads as neutral.")

    uncertain = norm_conf < uncertain_below and final == norm_label
    if uncertain:
        reasons.append(f"Low confidence ({norm_conf*100:.0f}%) - treat this prediction with caution.")

    token_scores, top_features = explain(model, vectorizer, cleaned) if with_explanation else (None, None)

    return {
        "raw": raw_text,
        "normalized": normalized,
        "cleaned": cleaned,
        "changes": changes,
        "masked": masked,
        "base_label": base_label, "base_conf": base_conf, "base_probs": base_probs,
        "model_label": norm_label, "model_conf": norm_conf, "model_probs": norm_probs,
        "final": final,
        "lean": lean,
        "reasons": reasons,
        "sarcasm": sarcastic, "sarcasm_evidence": sarcasm_evidence, "litotes": litotes,
        "strong_hedge": strong_hedge, "weak_hedge": weak_hedge,
        "aspects": aspects, "mixed": is_mixed,
        "uncertain": uncertain,
        "token_scores": token_scores, "top_features": top_features,
    }


# ---------------------------------------------------------------------------
# Challenge set used by the Stress Test tab
# ---------------------------------------------------------------------------

CHALLENGES = [
    ("Clear positive", "This product is absolutely amazing! I love it.", "Positive"),
    ("Clear negative", "This service is terrible and I am very disappointed.", "Negative"),
    ("Neutral fact", "The meeting is scheduled for tomorrow at 10 AM.", "Neutral"),
    ("Mixed opinion", "The phone has an excellent camera, but the battery life is disappointing.", "Mixed"),
    ("Slang + emoji", "This update is fire 🔥 absolutely loving it!", "Positive"),
    ("Negation", "The movie was not bad at all; I actually enjoyed it.", "Positive"),
    ("Sarcasm", "Great job! Another two-hour delay. Just what I needed.", "Negative"),
    ("Hidden complaint", "It works, I guess, but I wouldn't recommend buying it.", "Negative"),
    ("Sarcasm + emoji", "Oh great, my flight got cancelled again 🙄", "Negative"),
    ("Hinglish positive", "Yeh phone ekdum mast hai, camera zabardast 🔥", "Positive"),
    ("Hinglish negative", "Bilkul bakwas service, paisa barbad", "Negative"),
    ("Mixed (delivery)", "Delivery was super fast but the packaging was badly damaged.", "Mixed"),
    ("False-alarm check", "Loved the concert, can't wait to go again!", "Positive"),
]
