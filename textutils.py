"""Text preprocessing shared by train.py and app.py (so train/serve never drift apart)."""
import re

# NLTK's English stop-word list (179 words), embedded so training and serving always use the
# identical list and no corpus download is needed. (Same list as nltk.corpus.stopwords.)
STOP = set("""i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves
he him his himself she she's her hers herself it it's its itself they them their theirs themselves what which
who whom this that that'll these those am is are was were be been being have has had having do does did doing
a an the and but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own same so than too very s t
can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't""".split())

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")

# Publisher give-aways that let a model "cheat" without reading the content
# (e.g. ISOT real articles start with "WASHINGTON (Reuters) -"). Removing them
# makes the reported accuracy reflect language, not boilerplate.
_DATELINE = re.compile(r"^[^.]{0,60}\((reuters|ap|afp)\)\s*[-–—]?\s*", re.I)
_SOURCES = re.compile(r"\b(reuters|associated press|getty images|featured image|21st century wire|"
                      r"read more|share this|subscribe|photo by)\b", re.I)


def scrub_source_artifacts(text: str) -> str:
    text = _DATELINE.sub("", str(text).strip())
    return _SOURCES.sub(" ", text)


def clean_text(text: str) -> str:
    """lowercase -> drop URLs -> keep letters only -> remove stop-words."""
    text = _URL.sub(" ", str(text).lower())
    text = _NON_ALPHA.sub(" ", text)
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)
