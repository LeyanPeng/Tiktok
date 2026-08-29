"""Read-only in-memory catalog: indexing, bucketing, and priors.

Hard constraint: this module — and all of `src/` — must never import from
`evaluator/`. The organiser runs their own harness, where our copy of the
evaluator does not exist. The category rule is therefore reimplemented here
independently, and `tools/rule_parity.py` verifies on every run that the
reimplementation matches the official one across all 50,000 products.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Separator used when concatenating a bucket into one string. A newline
# guarantees we never synthesise a span that straddles two products.
SEPARATOR = "\n"

# Top-level category names the official `coarse_category` strips out.
EXCLUDED_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}

# Fields that participate in retrieval. This order is also the order in
# `Catalog.fields`.
TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")

# Attribute vocabularies. Deliberately the same word lists the official
# `classify_constraint` uses, so the attribute we reason about and the attribute
# the simulator reasons about are the same thing.
ATTRIBUTE_VOCAB: dict[str, tuple[str, ...]] = {
    "material": ("cotton", "polyester", "nylon", "leather", "wool",
                 "spandex", "silk", "rayon", "fabric"),
    "color": ("color", "colour", "black", "white", "blue", "red", "pink", "green",
              "brown", "gray", "grey", "purple", "yellow", "orange"),
    "size": ("size", "sizing", "width", "wide", "narrow"),
    "style": ("department", "style", "fit", "sleeve", "neck"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work"),
}
ATTRIBUTE_RE = {
    name: re.compile(r"\b(" + "|".join(words) + r")\b")
    for name, words in ATTRIBUTE_VOCAB.items()
}
PRICE_RE = re.compile(r"(?:\$|<=|under)\s*\d")

# How many constraints a customer states in one session. The simulator draws the
# same number of spans from the product's own copy.
CONSTRAINTS_PER_SESSION = 4

MATERIAL_FIRST_RE = re.compile(r"\b(" + "|".join(ATTRIBUTE_VOCAB["material"]) + r")\b")
COLOR_FIRST_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b"
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "need", "have", "has", "was", "were", "will", "can", "am",
}


def spans_of(value: object) -> list[str]:
    """Return a field's individual spans, preserving their boundaries.

    `flatten()` collapses a field into one string, which destroys span
    boundaries. The simulator slices the customer's stated requirements at
    exactly those boundaries, so any statistic about "what kinds of constraints
    customers state" has to be computed at span granularity — which is why it is
    computed during catalog load, while the original structure still exists.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def constraint_candidates(product: dict, corpus: str) -> list[str]:
    """The spans of this product that could become stated constraints.

    The simulator does not sample product copy uniformly. It lifts a material
    mention to position 0 and a colour mention to position 1, then appends
    features and details and keeps the first few. So the distribution of
    *constraint types* differs from the raw distribution of spans in product
    copy: material is systematically over-represented.

    Replaying that selection across the entire catalog yields the population
    distribution of constraint types rather than a sample from 200 sessions. The
    private split draws different products but shares this catalog and this
    selection logic, so the population estimate holds for both.

    This mirrors the evaluator's `intent_card`; `tools/prior_audit.py` verifies
    the resulting distribution still matches what the public sessions show.
    """
    spans: list[str] = []
    for field in ("features", "details"):
        spans.extend(spans_of(product.get(field)))

    material = MATERIAL_FIRST_RE.search(corpus)
    color = COLOR_FIRST_RE.search(corpus)
    if material:
        spans.insert(0, material.group(1))
    if color:
        spans.insert(1, f"color: {color.group(1)}")
    if product.get("price") not in (None, ""):
        spans.append(f"budget around ${product['price']}")

    cleaned = list(dict.fromkeys(
        re.sub(r"\s+", " ", s).strip(" -;,.")[:180] for s in spans
    ))
    return [s for s in cleaned if len(s) > 2][:CONSTRAINTS_PER_SESSION]


def classify_span(value: str) -> str:
    """Assign a span of product copy to an attribute type. `feature` is the catch-all."""
    low = value.lower()
    if "budget" in low or PRICE_RE.search(low):
        return "budget"
    for name in ("material", "color", "size", "style", "use_case"):
        if ATTRIBUTE_RE[name].search(low):
            return name
    return "feature"


def flatten(value: object) -> str:
    """Collapse a dict / list / scalar into one searchable string."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def coarse_category(values: list[str]) -> str:
    """Independent implementation of the official rule: last two path levels,
    top-level umbrella categories removed.

    Aligned character-for-character with `evaluator.local_evaluator.coarse_category`;
    `tools/rule_parity.py` verifies that across all 50,000 products.
    """
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in EXCLUDED_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class Catalog:
    """The read-only in-memory catalog.

    Builds once in under a second for 50,000 products, with no external
    dependency, no network, and no vector store.
    """

    def __init__(self, path: str | Path = "data/catalog.jsonl") -> None:
        self.path = Path(path)
        self.asins: list[str] = []
        self.blob: dict[str, str] = {}          # lowercased full text, for verbatim matching
        self.tokens: dict[str, set[str]] = {}   # deduplicated token set, for IDF coverage
        self.title: dict[str, str] = {}
        self.bucket: dict[str, list[str]] = defaultdict(list)
        self.bucket_of: dict[str, str] = {}
        self.idf: dict[str, float] = {}
        self.prior: dict[str, float] = {}       # popularity prior, tie-breaks only
        self.fields: dict[str, tuple[str, ...]] = {}   # per-field text, in TEXT_FIELDS order
        # Constraint-type tallies over product copy. Feeds the clarification prior
        # in `src/askpolicy.derive_type_prior`, replacing what used to be a
        # hardcoded table measured from the public session set.
        self.span_types: Counter[str] = Counter()
        self._bucket_blob: dict[str, str] = {}         # see bucket_blob()
        self._popular: dict[str, list[str]] = {}       # see top_popular()
        self._load()

    def _load(self) -> None:
        document_freq: dict[str, int] = defaultdict(int)
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                asin = str(product["parent_asin"])
                parts = tuple(flatten(product.get(f)).lower() for f in TEXT_FIELDS)
                text = " ".join(parts)

                self.asins.append(asin)
                self.blob[asin] = text
                self.fields[asin] = parts
                self.title[asin] = str(product.get("title") or "")

                name = coarse_category([str(v) for v in (product.get("categories") or [])])
                self.bucket[name].append(asin)
                self.bucket_of[asin] = name

                # Popularity prior: log review count x star rating. Under zero
                # information the best guess is the item most people bought,
                # which is exactly the item-popularity recommender baseline.
                count = product.get("rating_number") or 0
                stars = product.get("average_rating") or 0
                try:
                    self.prior[asin] = math.log1p(float(count)) * (float(stars) / 5.0)
                except (TypeError, ValueError):
                    self.prior[asin] = 0.0

                # Constraint-type tally. This can only happen here: once the loop
                # exits, the original JSON structure has been collapsed into one
                # string and the span boundaries no longer exist.
                for span in constraint_candidates(product, text):
                    self.span_types[classify_span(span)] += 1

                token_set = set(tokenize(text))
                self.tokens[asin] = token_set
                for token in token_set:
                    document_freq[token] += 1

        total = len(self.asins)
        self.idf = {t: math.log(total / (1 + c)) for t, c in document_freq.items()}
        # Longest first, for longest-prefix category matching.
        self.categories_by_length = sorted(self.bucket, key=len, reverse=True)

    def __len__(self) -> int:
        return len(self.asins)

    def candidates(self, category: str | None) -> list[str]:
        """Category pruning. Target containment measured at 200/200, which is what
        makes this the one hard filter the system allows itself."""
        if category and category in self.bucket:
            return self.bucket[category]
        return self.asins

    def bucket_blob(self, category: str | None) -> str:
        """Concatenate a whole category bucket into one cached string.

        A verbatim lookup used to walk 184 products in Python. Against one
        concatenated string it becomes a single C-level substring search. That
        optimisation is what makes "recover constraint spans by searching the
        catalog" fast enough to run at all.

        The newline separator guarantees no span is ever matched across a product
        boundary.
        """
        key = category or "__ALL__"
        cached = self._bucket_blob.get(key)
        if cached is None:
            cached = SEPARATOR.join(self.blob[a] for a in self.candidates(category))
            self._bucket_blob[key] = cached
        return cached

    def top_popular(self, category: str | None, k: int = 10) -> list[str]:
        """The k most popular items in a category, cached.

        Used by the failure fallback, which is why it must be fast. It originally
        sorted the candidate pool on every call — and before a category is known
        that meant sorting all 50,000 products. A fallback path should be the
        fastest path in the system, not the slowest.
        """
        key = category or "__ALL__"
        cached = self._popular.get(key)
        if cached is None:
            cached = sorted(
                self.candidates(category), key=lambda a: -self.prior.get(a, 0.0)
            )[:64]
            self._popular[key] = cached
        return cached[:k]

    def contains_verbatim(self, needle: str, category: str | None = None) -> bool:
        """Whether this text occurs verbatim in the candidate pool's product copy."""
        text = needle.strip().lower()
        if len(text) < 4:
            return False
        return text in self.bucket_blob(category)
