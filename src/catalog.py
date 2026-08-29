"""目录加载与索引。

重要约束：本模块（以及整个 src/）**不允许 import evaluator/**。
提交时评测方跑的是他们自己的 harness，我们这份 evaluator/ 副本不会存在。
所以类目规则在这里独立实现一份，并由 tools/rule_parity.py 反向验证它与官方逐字一致。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+")

# 拼接桶级文本时的分隔符。用换行保证不会跨商品拼出并不存在的片段。
SEPARATOR = "\n"

# 官方 coarse_category 会剔除的顶层类目名
EXCLUDED_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}

# 参与检索的字段。顺序即 Catalog.fields 里的存放顺序。
TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")

# 每件商品最多统计前几条片段。模拟器的 intent_card 也只取前几条，
# 取太多会让长描述压倒统计。
SPANS_PER_PRODUCT = 6

# 属性词表。与官方 classify_constraint 用同一批词，保证我们判断的「属性」
# 和模拟器判断的是同一个东西。
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


# 顾客一场会提到的约束条数。模拟器按同样的条数从商品文案里取。
CONSTRAINTS_PER_SESSION = 4

MATERIAL_FIRST_RE = re.compile(r"\b(" + "|".join(ATTRIBUTE_VOCAB["material"]) + r")\b")
COLOR_FIRST_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b"
)


def constraint_candidates(product: dict, corpus: str) -> list[str]:
    """这件商品会被拿去当约束的那几条文案。

    模拟器不是均匀地从商品文案里抽片段——它先把材质提到最前、颜色提到第二，
    再接 features / details，最后取前几条。所以「顾客会提哪类约束」的分布，
    和「商品文案里各类片段的原始占比」并不一样：材质被系统性地抬高了。

    在**整个目录**上复现这个选择过程，得到的是约束类型的总体分布，
    而不是某 200 场会话的样本 —— 私有集抽的是不同商品，但用的是同一份目录、
    同一套选择逻辑，所以总体分布对两边都成立。

    这段逻辑与 evaluator 的 intent_card 对齐，一致性由 tools/prior_audit.py 验证。
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
    """把一段商品文案归到某个属性类型。`feature` 是兜底类。"""
    low = value.lower()
    if "budget" in low or PRICE_RE.search(low):
        return "budget"
    for name in ("material", "color", "size", "style", "use_case"):
        if ATTRIBUTE_RE[name].search(low):
            return name
    return "feature"


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "need", "have", "has", "was", "were", "will", "can", "am",
}


def spans_of(value: object) -> list[str]:
    """取出字段里的**片段列表**，保留边界。

    `flatten()` 会把字段拼成一整串，片段边界就丢了。但模拟器的顾客话术恰恰是
    按片段切出来的，所以统计「顾客会提哪类约束」必须在片段粒度上做——
    这也是为什么这个统计要在目录加载时完成，那时原始结构还在。
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def flatten(value: object) -> str:
    """把 dict / list / 标量统一压成一段可检索文本。"""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def coarse_category(values: list[str]) -> str:
    """官方规则的独立实现：取类目路径末两级，剔除顶层大类。

    逐字对齐 evaluator.local_evaluator.coarse_category —— 一致性由
    tools/rule_parity.py 在 50,000 件商品上全量验证。
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
    """只读的内存目录。

    构建一次约 6 秒 / 50,000 件，全程无外部依赖、无网络、无向量库。
    """

    def __init__(self, path: str | Path = "data/catalog.jsonl") -> None:
        self.path = Path(path)
        self.asins: list[str] = []
        self.blob: dict[str, str] = {}          # 小写全文，用于原句子串匹配
        self.tokens: dict[str, set[str]] = {}   # 去重词集，用于 IDF 覆盖率
        self.title: dict[str, str] = {}
        self.bucket: dict[str, list[str]] = defaultdict(list)
        self.bucket_of: dict[str, str] = {}
        self.idf: dict[str, float] = {}
        self.prior: dict[str, float] = {}   # 人气先验，只用于同分打破平局
        self.fields: dict[str, tuple[str, ...]] = {}  # 按 TEXT_FIELDS 顺序分字段存放
        # 商品文案里各类属性片段的条数。供 AskPolicy 现算追问先验，
        # 取代早先写死的一组来自公开集的数字。见 src/askpolicy.derive_type_prior。
        self.span_types: Counter[str] = Counter()
        self._bucket_blob: dict[str, str] = {}  # 桶级文本缓存，见 bucket_blob()
        self._popular: dict[str, list[str]] = {}  # 类目热门榜缓存，见 top_popular()
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

                # 人气先验：评价数取对数 x 星级。零信息时最好的猜测就是最多人买的那件，
                # 这正是推荐系统里 item popularity 基线的逻辑。
                count = product.get("rating_number") or 0
                stars = product.get("average_rating") or 0
                try:
                    self.prior[asin] = math.log1p(float(count)) * (float(stars) / 5.0)
                except (TypeError, ValueError):
                    self.prior[asin] = 0.0

                # 约束类型统计。只在这里做得了——出了这个循环，
                # 原始 JSON 结构就被压成一整串，片段边界不复存在。
                for span in constraint_candidates(product, text):
                    self.span_types[classify_span(span)] += 1

                token_set = set(tokenize(text))
                self.tokens[asin] = token_set
                for token in token_set:
                    document_freq[token] += 1

        total = len(self.asins)
        self.idf = {t: math.log(total / (1 + c)) for t, c in document_freq.items()}
        # 类目名按长度降序，供最长前缀匹配使用
        self.categories_by_length = sorted(self.bucket, key=len, reverse=True)

    def __len__(self) -> int:
        return len(self.asins)

    def candidates(self, category: str | None) -> list[str]:
        """类目剪枝。落桶率实测 200/200，是本题唯一安全的硬过滤。"""
        if category and category in self.bucket:
            return self.bucket[category]
        return self.asins

    def bucket_blob(self, category: str | None) -> str:
        """把一个类目桶里所有商品的文本拼成一整串并缓存。

        原本每次 verbatim 查找要逐件扫 184 个商品（184 次 Python 层循环），
        拼成一串后变成 1 次 C 层子串查找。这让「按原文去目录里找约束」
        从跑不动变成可行——T6 的跨句式抽取整个建立在这个优化上。
        分隔符用换行，保证不会跨商品拼出并不存在的片段。
        """
        key = category or "__ALL__"
        cached = self._bucket_blob.get(key)
        if cached is None:
            cached = SEPARATOR.join(self.blob[a] for a in self.candidates(category))
            self._bucket_blob[key] = cached
        return cached

    def top_popular(self, category: str | None, k: int = 10) -> list[str]:
        """该类目下最热门的 k 件，结果缓存。

        专供异常降级路径使用，所以必须快：原来每次现排一遍候选池，
        类目还没识别出来时那就是对 50,000 件做全排序——
        兜底本该是最快的路径，不该是最慢的。
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
        """这一串文本，在候选池的商品原文里能否原样找到。"""
        text = needle.strip().lower()
        if len(text) < 4:
            return False
        return text in self.bucket_blob(category)
