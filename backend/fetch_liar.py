"""
Download the LIAR dataset (Wang, 2017, "Liar, Liar Pants on Fire") into data/liar/ and verify it.

    python fetch_liar.py

Tries the original UCSB archive first, then falls back to a GitHub mirror (byte-identical copies were found
in two independent mirrors). The files are checked against the published sizes: 12,836 statements in total
(train 10,269 / valid 1,284 / test 1,283) and the six expected labels. Standard library only.
"""
import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DEST = Path("data/liar")
EXPECTED = {"train": 10269, "valid": 1284, "test": 1283}
LABELS = {"pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"}
ORIGINAL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
MIRROR = "https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/master/liar_dataset/{}.tsv"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def from_original():
    z = zipfile.ZipFile(io.BytesIO(get(ORIGINAL)))
    return {s: z.read(f"{s}.tsv") for s in EXPECTED}


def from_mirror():
    return {s: get(MIRROR.format(s)) for s in EXPECTED}


def verify(files):
    for split, raw in files.items():
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8")), delimiter="\t", quoting=csv.QUOTE_NONE))
        labels = {r[1] for r in rows}
        if len(rows) != EXPECTED[split] or not labels <= LABELS or any(len(r) < 13 for r in rows):
            raise ValueError(f"{split}.tsv failed verification: {len(rows)} rows (expected {EXPECTED[split]}), labels {labels}")


if __name__ == "__main__":
    for name, fn in (("UCSB original", from_original), ("GitHub mirror", from_mirror)):
        try:
            files = fn()
            verify(files)
            DEST.mkdir(parents=True, exist_ok=True)
            for split, raw in files.items():
                (DEST / f"{split}.tsv").write_bytes(raw)
            print(f"LIAR downloaded from {name} and verified -> {DEST}/ ({sum(EXPECTED.values()):,} statements)")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} failed: {type(e).__name__}: {str(e)[:120]}")
    sys.exit("Could not download LIAR. Get liar_dataset.zip from the UCSB link above and unzip it into data/liar/")
