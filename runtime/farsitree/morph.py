"""Unsupervised Persian suffix (and, in version 2, verb prefix) splitting for Harraf's tokenizer.

Persian puts much of the grammar in word endings: plural ‌ها, ezafe/indefinite ی,
verb person endings م ی یم ید ند, possessive clitics ش مان تان شان.  A BPE
tokenizer merges them with the stem inconsistently, so «رفت» and «رفتند» can
end up as unrelated tokens and agreement has to be learned word by word.

Here a suffix is split off only when the remaining stem is itself a frequent
word of the corpus, so «رفتند» -> «رفت +ند» but «چند» stays whole.  This needs
no dictionary and works on real text.  The «+» marks a bound suffix and is
removed again when decoding.
"""
import json
from collections import Counter

ZWNJ = "‌"
# longest first; ZWNJ forms are written with the joiner
SUFFIXES = [ZWNJ + "هایی", ZWNJ + "های", ZWNJ + "ها", "هایی", "های", "ها", "ترین", "تر",
            "شان", "تان", "مان", "ند", "یم", "ید", "یی", "ای", "م", "ی", "ش", "ت"]
MARK = "+"

# v2: verb prefixes, and the present-tense endings that only count after one of them
# («می‌رود» -> «می‌+ رو +د»; a bare «د» ending is never split, since «بود» is not «بو +د»)
PREFIXES = ["نمی" + ZWNJ, "می" + ZWNJ]
VERB_SUFFIXES = ["ند", "یم", "ید", "د", "م", "ی"]


class Segmenter:
    """version 1: suffixes only (Harraf v1). version 2: also «می‌»/«نمی‌» and present-tense endings."""

    def __init__(self, stems, min_stem=2, version=1, verb_stems=()):
        self.stems = set(stems)
        self.min_stem = min_stem
        self.version = version
        self.verb_stems = set(verb_stems)
        self.cache = {}

    @classmethod
    def fit(cls, texts, min_count=3, version=1):
        counts = Counter(w for t in texts for w in t.split())
        verb_stems = set()
        if version >= 2:
            # a present stem is real when it shows up after «می‌/نمی‌» with two or more different endings
            endings = {}
            for w in counts:
                for pre in PREFIXES:
                    if w.startswith(pre):
                        rest = w[len(pre):]
                        for suf in VERB_SUFFIXES:
                            if rest.endswith(suf) and len(rest) - len(suf) >= 1:
                                endings.setdefault(rest[:-len(suf)], set()).add(suf)
                        break
            verb_stems = {st for st, e in endings.items() if len(e) >= 2}
        return cls([w for w, c in counts.items() if c >= min_count], version=version, verb_stems=verb_stems)

    def split_word(self, w):
        if w in self.cache:
            return self.cache[w]
        if self.version >= 2:
            for pre in PREFIXES:
                if w.startswith(pre) and len(w) > len(pre):
                    rest = w[len(pre):]
                    for suf in VERB_SUFFIXES:
                        if rest.endswith(suf) and rest[:-len(suf)] in self.verb_stems:
                            out = [pre + MARK, rest[:-len(suf)], MARK + suf]
                            break
                    else:
                        out = [pre + MARK, rest]
                    self.cache[w] = out
                    return out
        out = [w]
        for suf in SUFFIXES:
            stem = w[:-len(suf)]
            if w.endswith(suf) and len(stem) >= self.min_stem and stem in self.stems:
                rest = self.split_word(stem)                      # «کتاب‌هایش» -> کتاب +‌های +ش
                out = rest + [MARK + suf]
                break
        self.cache[w] = out
        return out

    def __call__(self, text):
        return " ".join(p for w in text.split() for p in self.split_word(w))

    @staticmethod
    def join(text):
        return text.replace(" " + MARK, "").replace(MARK + " ", "")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            if self.version == 1:   # v1 file format, unchanged
                json.dump(sorted(self.stems), f, ensure_ascii=False)
            else:
                json.dump({"version": self.version, "stems": sorted(self.stems), "verb_stems": sorted(self.verb_stems)},
                          f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, list):
            return cls(d)
        return cls(d["stems"], version=d["version"], verb_stems=d["verb_stems"])
