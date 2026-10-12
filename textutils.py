"""
Text preprocessing shared by train.py and server.py.

There is exactly ONE entry point, `preprocess(text)`, and both training and inference call it.
`PREPROCESS_VERSION` is saved with the trained artefacts and checked by server.py at start-up, so a
model trained with one version of this file can never silently be served with another.

What changed in v2 (vs. v1) and why — short claims have very few words, so every word counts:
  * Negations are KEPT (no, nor, not, never, none, nobody, nothing, nowhere, neither, cannot, without).
    v1 deleted "not"/"no"/"nor" as stop-words, so "Tax cuts did not create jobs" and
    "Tax cuts did create jobs" became the same string. Contractions are expanded first
    (don't -> do not, won't -> will not, can't -> can not) so that "n't" is not lost either.
  * A few quantifier/comparison words that change a claim's meaning are KEPT as well
    (all, any, few, more, most, only, than, against).
  * Numbers are no longer deleted. Claims are full of figures ("30 percent", "$2 million", "in 2009"),
    so numbers become class tokens (_pct_, _money_, _year_, _num_) instead of vanishing —
    the model can learn "a precise percentage" without memorising individual values.
  * "!" and "?" become tokens (_excl_, _ques_) instead of being stripped with other punctuation.
  * Everything else (URLs, other punctuation, symbols) is still removed; text is lower-cased.
"""
import re

PREPROCESS_VERSION = "2"

# NLTK's English stop-word list (179 words), embedded so training and serving use the identical list
# and no corpus download is needed. (Same list as nltk.corpus.stopwords.)
_NLTK_STOP = set("""i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves what which
who whom this that that'll these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own same so than too very s t
can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split())

NEGATIONS = {"no", "nor", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "cannot", "without"}
QUANTIFIERS = {"all", "any", "few", "more", "most", "only", "than", "against"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTRACTIONS = [(re.compile(p, re.I), r) for p, r in [
    (r"\bcan['’]t\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon['’]t\b", "will not"),
    (r"\bshan['’]t\b", "shall not"), (r"\bain['’]t\b", "is not"),
    (r"(\w)n['’]t\b", r"\1 not"),                      # don't, isn't, wouldn't ...
    (r"['’](s|re|ve|ll|d|m)\b", ""),                   # possessives / other contractions
]]
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion|thousand|[mbk])\b)?", re.I)
_PCT = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:%|percent\b|per cent\b)", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_KEEP = re.compile(r"[^a-z_\s]")

# Publisher give-aways that let a model "cheat" without reading the content
# (e.g. ISOT real articles start with "WASHINGTON (Reuters) -"). Removing them
# makes the reported accuracy reflect language, not boilerplate.
_DATELINE = re.compile(r"^[^.]{0,60}\((reuters|ap|afp)\)\s*[-–—]?\s*", re.I)
_SOURCES = re.compile(r"\b(reuters|associated press|getty images|featured image|21st century wire|"
                      r"read more|share this|subscribe|photo by)\b", re.I)


def scrub_source_artifacts(text: str) -> str:
    text = _DATELINE.sub("", str(text).strip())
    return _SOURCES.sub(" ", text)


def make_cleaner(keep_negations=True, keep_quantifiers=True, numbers="tokens", punct_tokens=True):
    """Build a cleaning function. The production default is `clean_text`; the options exist so the
    choices above can be ablated on validation data (see train.py --ablate-preprocessing)."""
    stop = set(_NLTK_STOP)
    if keep_negations:
        stop -= NEGATIONS
    if keep_quantifiers:
        stop -= QUANTIFIERS

    def clean(text: str) -> str:
        t = _URL.sub(" ", str(text))
        if keep_negations:
            for pat, rep in _CONTRACTIONS:
                t = pat.sub(rep, t)
        t = t.lower()
        if numbers == "tokens":
            t = _MONEY.sub(" _money_ ", t)
            t = _PCT.sub(" _pct_ ", t)
            t = _YEAR.sub(" _year_ ", t)
            t = _NUM.sub(" _num_ ", t)
        if punct_tokens:
            t = t.replace("!", " _excl_ ").replace("?", " _ques_ ")
        t = _KEEP.sub(" ", t)
        return " ".join(w for w in t.split() if w not in stop and (len(w) > 1 or w == "_"))
    return clean


clean_text = make_cleaner()


def preprocess(text: str) -> str:
    """THE preprocessing used for training and for inference."""
    return clean_text(scrub_source_artifacts(text))
