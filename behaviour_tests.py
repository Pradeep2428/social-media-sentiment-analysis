"""
Behaviour test suite for SentiLens, in the style of CheckList
(Ribeiro et al., "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList", ACL 2020).

Instead of one accuracy number, each *capability* gets its own set of generated test
posts and its own pass rate. Three kinds of test are used:

  MFT  Minimum Functionality Test - each post has an expected label
  INV  Invariance test            - small changes (brand name, typo) must not change the label

The posts are generated from templates and word lists written for this suite,
independently of the rules in smart_layer.py. The rules were not tuned to these tests.
"""

import itertools
import random

SEED = 42
PER_TEST = 40  # posts sampled per capability

PRODUCTS = ["phone", "laptop", "app", "camera", "headphones", "service", "update",
            "movie", "food", "hotel", "course", "game"]
POS_ADJ = ["amazing", "excellent", "great", "fantastic", "wonderful", "brilliant",
           "superb", "awesome", "good", "lovely", "impressive", "perfect"]
NEG_ADJ = ["terrible", "awful", "horrible", "bad", "disappointing", "useless",
           "poor", "pathetic", "annoying", "frustrating", "rubbish", "mediocre"]
EVENTS = ["crashed", "stopped working", "broke", "froze", "died", "got cancelled",
          "arrived late", "lost all my data", "overheated", "logged me out"]
ASPECTS = ["camera", "battery", "screen", "price", "sound", "design", "delivery",
           "packaging", "staff", "food"]
HING_POS = ["mast", "badhiya", "zabardast", "shandar", "kamaal", "accha"]
HING_NEG = ["bekar", "bakwas", "ghatiya", "faltu", "kharab", "bura"]
EMOJI_POS = ["😍", "🔥", "🥰", "👏", "💯"]
EMOJI_NEG = ["😡", "🤬", "👎", "😤", "💔"]
BRANDS = ["Amazon", "Flipkart", "Zomato", "Swiggy", "Nvidia", "Microsoft", "Google",
          "Paytm", "Netflix", "Ola"]
DAYS = ["Monday", "Tuesday", "Friday", "the weekend", "next week", "the 15th"]


def _fill(templates, **slots):
    """Every combination of templates x slot values."""
    out = []
    keys = list(slots)
    for t in templates:
        used = [k for k in keys if "{" + k + "}" in t]
        for combo in itertools.product(*[slots[k] for k in used]):
            out.append(t.format(**dict(zip(used, combo))))
    return out


def _sample(items, n=PER_TEST, seed=SEED):
    items = sorted(set(items))
    rng = random.Random(seed)
    return items if len(items) <= n else rng.sample(items, n)


def _typo(text, rng):
    """Swap two neighbouring letters inside one random longer word."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return text
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


VOCAB_POS = ["The {p} is {a}.", "This {p} is really {a}.", "Honestly, the {p} was {a}.",
             "I think the {p} is {a}."]
VOCAB_NEG = VOCAB_POS


def _build_dev():
    """Development split: used while improving the rules."""
    suite = []

    def mft(name, desc, posts, expected, example=None):
        posts = _sample(posts)
        suite.append({"name": name, "kind": "MFT", "desc": desc, "posts": posts,
                      "expected": expected, "example": example or posts[0]})

    mft("Clear praise", "Plain positive words should read as Positive.",
        _fill(VOCAB_POS, p=PRODUCTS, a=POS_ADJ), {"Positive"})
    mft("Clear criticism", "Plain negative words should read as Negative.",
        _fill(VOCAB_NEG, p=PRODUCTS, a=NEG_ADJ), {"Negative"})
    mft("Neutral facts", "Statements with no opinion should read as Neutral.",
        _fill(["The {p} launches on {d}.", "I am picking up the {p} on {d}.",
               "The {p} review goes live on {d}."], p=PRODUCTS, d=DAYS), {"Neutral"})
    mft("Negated praise", "“Not good” should not read as Positive.",
        _fill(["The {p} is not {a}.", "This {p} isn't {a} at all.", "I don't think the {p} is {a}."],
              p=PRODUCTS, a=POS_ADJ), {"Negative", "Neutral"})
    mft("Negated criticism", "“Not bad” should not read as Negative.",
        _fill(["The {p} is not {a}.", "This {p} isn't {a} at all.", "Honestly, the {p} was not {a}."],
              p=PRODUCTS, a=NEG_ADJ), {"Positive", "Neutral"})
    mft("Sarcasm", "Praise wrapped around a bad experience should read as Negative.",
        _fill(["Oh great, the {p} {e} again.", "Wow, love it when my {p} {e}.",
               "Fantastic, the {p} {e}. Just perfect.", "Thanks a lot, my {p} {e} on day one."],
              p=PRODUCTS, e=EVENTS), {"Negative"})
    mft("Quiet complaints", "Polite dissatisfaction without harsh words should read as Negative.",
        _fill(["The {p} works, I guess, but I wouldn't recommend it.", "The {p} is okay, but not worth the price.",
               "Expected a lot more from this {p}.", "I'll be returning this {p}.",
               "Would not buy this {p} again.", "The {p} is fine I suppose, but I regret buying it."],
              p=PRODUCTS), {"Negative"})
    mft("Mixed opinions", "One good and one bad aspect should read as Mixed.",
        [t for t in _fill(["The {x} is {a}, but the {y} is {b}.", "The {y} is {b}, but the {x} is {a}."],
                          x=ASPECTS, y=ASPECTS, a=POS_ADJ[:6], b=NEG_ADJ[:6])
         if t.split()[1] != t.split()[-3]], {"Mixed"})
    mft("Hinglish praise", "Common Hinglish praise should read as Positive.",
        _fill(["{p} ekdum {h} hai", "Yeh {p} bahut {h} hai", "{p} toh {h} nikla yaar"],
              p=PRODUCTS, h=HING_POS), {"Positive"})
    mft("Hinglish criticism", "Common Hinglish criticism should read as Negative.",
        _fill(["{p} ekdum {h} hai", "Yeh {p} bahut {h} hai", "{p} toh {h} nikla yaar"],
              p=PRODUCTS, h=HING_NEG), {"Negative"})
    mft("Emoji tone", "An emoji should set the tone of an otherwise neutral post.",
        _fill(["Just got the new {p} {e}", "Finally tried the {p} {e}"], p=PRODUCTS, e=EMOJI_POS)
        + _fill(["Just got the new {p} {e}", "Finally tried the {p} {e}"], p=PRODUCTS, e=EMOJI_NEG),
        None)  # expected comes from the emoji, see run_suite

    # Invariance: the brand name should not change the verdict
    rng = random.Random(SEED)
    groups = []
    for a in POS_ADJ[:6] + NEG_ADJ[:6]:
        for t in ["I just tried {b} and it was {a}.", "My experience with {b} has been {a}."]:
            groups.append([t.format(b=b, a=a) for b in rng.sample(BRANDS, 5)])
    suite.append({"name": "Brand names", "kind": "INV",
                  "desc": "Swapping the brand (Amazon → Zomato) should not change the verdict.",
                  "groups": groups, "example": groups[0][0] + "  ↔  " + groups[0][1]})

    # Invariance: a small typo should not change the verdict
    base = _sample(_fill(VOCAB_POS, p=PRODUCTS, a=POS_ADJ) + _fill(VOCAB_NEG, p=PRODUCTS, a=NEG_ADJ), seed=7)
    rng = random.Random(SEED)
    groups = [[s, _typo(s, rng)] for s in base]
    groups = [g for g in groups if g[0] != g[1]]
    suite.append({"name": "Typos", "kind": "INV",
                  "desc": "A small spelling slip should not change the verdict.",
                  "groups": groups, "example": groups[0][0] + "  ↔  " + groups[0][1]})
    return suite


# ---------------------------------------------------------------------------
# Held-out split: different templates, products, adjectives, events, brands and
# Hinglish words. Written before the rules were improved and run only for the
# final report.
# ---------------------------------------------------------------------------

PRODUCTS_T = ["tablet", "smartwatch", "charger", "website", "subscription", "restaurant",
              "flight", "series", "keyboard", "earbuds", "printer", "show"]
POS_ADJ_T = ["outstanding", "incredible", "terrific", "delightful", "exceptional", "solid",
             "stunning", "marvellous", "top-notch", "enjoyable", "pleasant", "remarkable"]
NEG_ADJ_T = ["lousy", "atrocious", "subpar", "disgusting", "worthless", "defective",
             "unbearable", "shoddy", "flimsy", "overrated", "clunky", "laggy"]
EVENTS_T = ["shut down mid-call", "charged me twice", "went offline", "deleted my playlist",
            "missed the delivery window", "drained in an hour", "cancelled my booking",
            "failed to update", "ate my money", "kept buffering"]
ASPECTS_T = ["display", "speakers", "interface", "ambience", "fabric", "fit", "story",
             "music", "checkout", "support"]
HING_POS_T = ["badiya", "jhakaas", "behtareen", "lajawab", "kamal", "sahi"]
HING_NEG_T = ["bekaar", "bakwaas", "ghatia", "faaltu", "wahiyat", "kharaab"]
EMOJI_POS_T = ["🤩", "😁", "❤️", "✨", "🙌"]
EMOJI_NEG_T = ["😠", "😭", "🤮", "😞", "😩"]
BRANDS_T = ["Apple", "Samsung", "Facebook", "Verizon", "Uber", "Spotify", "Tesla", "Myntra",
            "Jio", "PlayStation"]
DAYS_T = ["Thursday", "Saturday", "next month", "the 3rd", "tomorrow morning", "Diwali"]


def _build_test():
    suite = []

    def mft(name, desc, posts, expected):
        posts = _sample(posts, seed=SEED + 1)
        suite.append({"name": name, "kind": "MFT", "desc": desc, "posts": posts,
                      "expected": expected, "example": posts[0]})

    vocab = ["Really happy with the {p}, it is {a}.", "The new {p}? {a}.", "Gotta say the {p} is {a}."]
    mft("Clear praise", "Plain positive words should read as Positive.",
        _fill(vocab, p=PRODUCTS_T, a=POS_ADJ_T), {"Positive"})
    mft("Clear criticism", "Plain negative words should read as Negative.",
        _fill(vocab, p=PRODUCTS_T, a=NEG_ADJ_T), {"Negative"})
    mft("Neutral facts", "Statements with no opinion should read as Neutral.",
        _fill(["The {p} will be available from {d}.", "Our team is discussing the {p} on {d}.",
               "Delivery of the {p} is scheduled for {d}."], p=PRODUCTS_T, d=DAYS_T), {"Neutral"})
    neg = ["The {p} was never {a}.", "Can't say the {p} is {a}.", "The {p} is far from {a}."]
    mft("Negated praise", "“Not good” should not read as Positive.",
        _fill(neg, p=PRODUCTS_T, a=POS_ADJ_T), {"Negative", "Neutral"})
    mft("Negated criticism", "“Not bad” should not read as Negative.",
        _fill(neg, p=PRODUCTS_T, a=NEG_ADJ_T), {"Positive", "Neutral"})
    mft("Sarcasm", "Praise wrapped around a bad experience should read as Negative.",
        _fill(["Brilliant, the {p} {e}. Love that for me.", "Just amazing how the {p} {e}.",
               "Best day ever, my {p} {e}.", "So glad the {p} {e} right before my exam."],
              p=PRODUCTS_T, e=EVENTS_T), {"Negative"})
    mft("Quiet complaints", "Polite dissatisfaction without harsh words should read as Negative.",
        _fill(["Honestly I expected better from this {p}.", "The {p} is alright but I want my money back.",
               "Not sure the {p} was worth it.", "I'm switching away from this {p}.",
               "Meh, the {p} is just okay, I won't be buying again.", "Can't really recommend the {p}."],
              p=PRODUCTS_T), {"Negative"})
    def two_aspects(t):
        return len({w for w in ASPECTS_T if w in t.replace(",", " ").replace(".", " ").split()}) == 2
    loved = [t for t in _fill(["Loved the {x}, hated the {y}.", "Hated the {y}, loved the {x}."],
                              x=ASPECTS_T, y=ASPECTS_T) if two_aspects(t)]
    although = [t for t in _fill(["The {x} was {a} although the {y} was {b}."],
                                 x=ASPECTS_T, y=ASPECTS_T, a=POS_ADJ_T[:6], b=NEG_ADJ_T[:6]) if two_aspects(t)]
    mixed = _sample(loved, 20, SEED + 1) + _sample(although, 20, SEED + 1)
    suite.append({"name": "Mixed opinions", "kind": "MFT", "desc": "One good and one bad aspect should read as Mixed.",
                  "posts": mixed, "expected": {"Mixed"}, "example": mixed[0]})
    mft("Hinglish praise", "Common Hinglish praise should read as Positive.",
        _fill(["{p} kya {h} hai yaar", "bhai {p} toh ekdum {h}", "{p} sach mein {h} hai"],
              p=PRODUCTS_T, h=HING_POS_T), {"Positive"})
    mft("Hinglish criticism", "Common Hinglish criticism should read as Negative.",
        _fill(["{p} kya {h} hai yaar", "bhai {p} toh ekdum {h}", "{p} sach mein {h} hai"],
              p=PRODUCTS_T, h=HING_NEG_T), {"Negative"})
    mft("Emoji tone", "An emoji should set the tone of an otherwise neutral post.",
        _fill(["Ordering the {p} again {e}", "Day one with the {p} {e}"], p=PRODUCTS_T, e=EMOJI_POS_T)
        + _fill(["Ordering the {p} again {e}", "Day one with the {p} {e}"], p=PRODUCTS_T, e=EMOJI_NEG_T),
        None)

    rng = random.Random(SEED + 1)
    groups = []
    for a in POS_ADJ_T[:6] + NEG_ADJ_T[:6]:
        for t in ["Ordered from {b} yesterday, the experience was {a}.", "{b} customer support was {a} today."]:
            groups.append([t.format(b=b, a=a) for b in rng.sample(BRANDS_T, 5)])
    suite.append({"name": "Brand names", "kind": "INV",
                  "desc": "Swapping the brand should not change the verdict.",
                  "groups": groups, "example": groups[0][0] + "  ↔  " + groups[0][1]})
    base = _sample(_fill(vocab, p=PRODUCTS_T, a=POS_ADJ_T) + _fill(vocab, p=PRODUCTS_T, a=NEG_ADJ_T), seed=SEED + 2)
    groups = [[s, _typo(s, rng)] for s in base]
    groups = [g for g in groups if g[0] != g[1]]
    suite.append({"name": "Typos", "kind": "INV",
                  "desc": "A small spelling slip should not change the verdict.",
                  "groups": groups, "example": groups[0][0] + "  ↔  " + groups[0][1]})
    return suite


def build_suite(split="dev"):
    """split='dev' is used while improving rules; split='test' is held out for the final report."""
    return _build_test() if split == "test" else _build_dev()


def _emoji_expected(post):
    return {"Positive"} if any(e in post for e in EMOJI_POS + EMOJI_POS_T) else {"Negative"}


def run_suite(analyze, split="dev"):
    """
    analyze(text) -> (base_label, smart_label)
    Returns one result row per capability with pass rates and example failures.
    """
    rows = []
    for cap in build_suite(split):
        if cap["kind"] == "MFT":
            base_pass = smart_pass = 0
            fails = []
            for post in cap["posts"]:
                exp = cap["expected"] or _emoji_expected(post)
                b, s = analyze(post)
                base_pass += b in exp
                smart_pass += s in exp
                if s not in exp:
                    fails.append((post, s, "/".join(sorted(exp))))
            n = len(cap["posts"])
            base_rate = None if cap["expected"] == {"Mixed"} else base_pass / n
            expected_txt = "/".join(sorted(cap["expected"])) if cap["expected"] else "Follows the emoji"
        else:
            base_pass = smart_pass = 0
            fails = []
            for g in cap["groups"]:
                outs = [analyze(p) for p in g]
                base_pass += len({o[0] for o in outs}) == 1
                same = len({o[1] for o in outs}) == 1
                smart_pass += same
                if not same:
                    fails.append((" ↔ ".join(g[:2]), " vs ".join(o[1] for o in outs[:2]), "same verdict"))
            n = len(cap["groups"])
            base_rate = base_pass / n
            expected_txt = "Same verdict"
        rows.append({
            "name": cap["name"], "kind": cap["kind"], "desc": cap["desc"], "example": cap["example"],
            "expected": expected_txt, "n": n, "base": base_rate, "smart": smart_pass / n,
            "fails": fails[:6], "fail_count": len(fails),
        })
    return rows
