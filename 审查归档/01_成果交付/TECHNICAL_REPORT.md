# Technical Report — Offline Shopping Copilot

**TechJam 2026 · Track 4 · Shopping Copilot: AI Conversational Search and Recommendations**
Solo entry · https://github.com/LeyanPeng/Tiktok

---

## Summary

| | Official baseline | This system |
| --- | ---: | ---: |
| Hit Rate@10 | 0.125 | **1.000** |
| MRR | 0.068034 | **0.674806** |
| MTTC | 9.81 | **1.565** |
| Efficiency | 0.119 | **0.9435** |
| **TechnicalScore** | **0.10671** | **0.891142** |
| Tokens | 0 | **0** |
| Network required | — | **No** |
| 200-session wall clock | — | **6.2 s** (0.031 s/session) |

Under a self-authored paraphrase perturbation of the customer simulator, the
system scores **0.813505** — still 7.6× the unperturbed baseline.

Every number in this report is produced by a command in the repository and
exits non-zero on failure. See §7.

---

## 1. The decision that shaped everything

`docs/submission_rules.md` states that **for official final scoring, organizer
policy may disable network access**, and requires each team to declare whether
their system needs it.

The problem statement's Pillar I prescribes a pipeline ending in *LLM Semantic
Ranking*. Any such pipeline requires a declared offline fallback to run under a
network cut-off. We took the constraint seriously and inverted the design
question: *how much of this task is actually about model capacity, and how much
is about the structure of the problem?*

Within the first hour we built a throwaway 130-line probe using only the Python
standard library. It scored **0.852** against a baseline of **0.107**. That
settled the architecture: **the entire scoring path runs offline**, and there is
no online path to fall back from.

This is the single highest-leverage decision in the project, and it is a
deliberate, documented deviation from the literal pipeline prescription. §3
explains the two other places we deviated and what evidence drove each.

---

## 2. Architecture, mapped to the four pillars

```
customer message
      │
 L0  Session state machine ──────── Pillar II
      │  incremental slot accumulation · intent override with weight decay
      │  turn budget · saturation detection · stall detection
      ▼
 L1  Dual-track intent routing ──── Pillar I
      │  Buying  → constraint present at turn 1 → rank aggressively
      │  Browsing → vague opener → popularity prior, spend a turn to learn
      ▼
 L2  Multi-route retrieval ──────── Pillar I
      │  R1 category-bucket pruning   50,000 → median 184   (only hard filter)
      │  R2 verbatim fingerprint match on constraint spans
      │  R3 IDF-weighted token coverage as graded fallback
      ▼
 L3  Ranking ────────────────────── Pillar I
      │  strictly additive · no conjunctive filtering · popularity tie-break
      ▼
 L4  Clarification policy ───────── Pillar II + III
      │  P(constraint type) × H(attribute | live candidate pool)
      │  runtime strategy switch when the ranking stops moving
      ▼
 message · ask_attribute · 10 ranked parent_asin
```

### Pillar I — Intent Routing & Hybrid Pipeline

**Dual-track routing.** The opening message is classified at turn 1. A *Buying*
opener carries a hard constraint, so the system ranks immediately and asks
sparingly. A *Browsing* opener carries nothing, so returning catalog-order
results wastes a turn; instead the track falls back to a popularity prior — the
classic item-popularity recommender baseline, which is the best available
estimate under zero information. Wiring this single behaviour lifted Hit
Rate@10 from 0.975 to **1.000** and repaired the boundary scenario from
hit 0.80 / MRR 0.372 to **1.00 / 0.476**.

**Multi-route retrieval.** Three routes, combined additively:

*R1 — category-bucket pruning.* The organiser derives the customer's opening
category phrase from the target product's own `categories` field. Reimplementing
that rule over the catalog yields 1,115 buckets, median size 184, a 271×
reduction. We measured the target's containment rate at **200/200**, which is
precisely what licenses using it as a hard filter. It is the only hard filter in
the system.

*R2 — verbatim fingerprint matching.* The simulated customer's requirements are
sliced verbatim out of the target's own `features` / `details` text. A phrase
such as `"Long torso camisole for extra coverage with spagetti adjustable
strap"` functions as a product fingerprint; substring-matching it against the
catalog is a stronger signal than any embedding available offline.

*R3 — IDF-weighted coverage.* When a span does not match verbatim, its terms
contribute proportionally to their inverse document frequency. Rare terms
matter more. This is a graded fallback, deliberately capped well below the
verbatim weight so a coincidental term overlap can never outrank a fingerprint.

**On "LLM Semantic Ranking".** We do not perform it, and this is a considered
choice rather than an omission. Three converging reasons: (a) network access may
be disabled at scoring time; (b) ProductAgent reports that in the *conversational*
setting dense retrieval collapses to HIT@10 8.27 against BM25's 39.48, because
queries synthesised from a dialogue consist largely of the customer's own
supplied terms — exactly the regime where lexical matching dominates; (c) our own
data confirms the regime: constraints are literal catalog substrings, so lexical
matching is not an approximation of the right answer, it *is* the right answer.
The ranking stage is semantic in intent and lexical in mechanism.

**On Reciprocal Rank Fusion.** ProductAgent reports RRF degrading results
consistently across both settings. We combine routes additively into a single
score rather than fusing separate ranked lists, and never measured a need for
fusion.

### Pillar II — Dialog Strategy

**Dynamic state machine.** Constraints accumulate as weighted slots carrying
their source turn. On detecting an intent override, prior slots are **decayed by
×0.35 rather than erased**. This is a deliberate departure from the literal
"slot erasure" language in the problem statement, and it is grounded in the data:
in this task *every* constraint originates from the same target product, so
erasing earlier slots discards evidence that still points at the correct answer.
Decay expresses "the new intent takes priority" without self-inflicted recall
loss. Intent-override sessions achieve **MRR 0.9048**, the strongest of the four
scenarios.

**Proactive guidance.** The system asks about exactly one attribute per turn, as
the API contract permits, chosen by expected information gain (Pillar III). It
stops asking when the customer has explicitly declined to express a preference on
three consecutive occasions.

An earlier version stopped after accumulating four constraints, four being the
number every public session holds. We replaced it because that constant encodes an
assumption about the private set that the specification never makes — the spec
fixes the *scenario mix*, not the constraint count — and the cost is asymmetric:
stopping early closes the information channel permanently, while asking too long
costs nothing measurable.

Honesty requires adding that the replacement **did not improve any score we could
measure**. Under a stress harness that widens sessions to six constraints, with and
without paraphrase, the two rules are identical to six decimal places
(`tools/constraint_count_stress.py`), because 71.5% of sessions convert on turn 1
and the gate rarely gets to act at all. The change buys the removal of an
unverified assumption, not points.

One genuine defect surfaced along the way. The first replacement counted a round
as barren whenever *extraction* returned nothing — conflating "the customer has
nothing more to say" with "we failed to parse what they said". Under paraphrase
those are exactly the rounds where parsing fails, so the agent fell silent early
and the hard regime lost 0.0046 (buying Hit Rate 0.975 → 0.963). Keying on the
customer's explicit refusal instead of our own parser's success restored parity.

**Honest note on clarification-question phrasing.** The evaluator's simulated
customer reads only the structured `ask_attribute` field; the natural-language
`message` never influences the score. We wrote careful phrasings for the demo
and for human readers, and spent no time tuning them for points.

### Pillar III — Self-Evolution

**Runtime adaptation.** Attribute priorities are recomputed every turn from the
**live candidate pool**, not from a fixed order. For each unasked attribute:

```
expected value = P(customer holds a constraint of this type)
               × H(attribute's value distribution across the current pool)
```

The left term is measured, not assumed: 800 constraints across 200 public
sessions distribute as feature 50.5%, material 37.8%, color 7.5%, remainder
4.3%. The right term is normalised entropy over the surviving candidates — an
attribute on which every candidate agrees carries zero information, and asking
about it burns a turn. This is the twenty-questions optimal-split criterion, and
it is the mechanism ProductAgent's ablation identifies as the largest single
lever in that system (HIT@10 15.60 → 47.00 when clarification is grounded in
live pool statistics rather than generated freely).

**Adaptive orchestration.** The system tracks whether its own top-5 changed
between turns. Two consecutive turns of a frozen ranking *after* information has
arrived means the current line of questioning is not converging; the policy then
switches selection criteria at runtime — discarding the prior term and choosing
purely by pool divergence. The guard "after information has arrived" was added
because an unchanged ranking at the very start means *the conversation has not
begun*, not that the strategy failed; without it, boundary sessions burned a turn
on a low-prior attribute (MTTC 1.565 → 1.575).

**We do not claim this pillar earned score on the public set.** §5 reports the
measurement honestly.

### Pillar IV — Evaluation

We treat the official metric as a budget to be allocated, not a number to be
admired:

```
TechnicalScore = 0.50·HitRate@10 + 0.30·MRR + 0.20·Efficiency
```

| Component | Current | Contribution | Headroom | Decision |
| --- | ---: | ---: | ---: | --- |
| Hit Rate@10 | 1.0000 | 0.5000 | 0.0000 | Saturated — invest nothing |
| MRR | 0.6748 | 0.2024 | **0.0976** | The entire remaining game |
| Efficiency | 0.9435 | 0.1887 | 0.0113 | Near structural floor |

Efficiency is close to its floor because intent-override sessions cannot convert
before turn 3–4 by rule; their MTTC of 3.6 is near-optimal. Once Hit Rate
saturated, **89% of all remaining points sat in MRR**, and every subsequent
engineering decision was tested against that single question. §5 records that we
attacked it five times and did not move it.

---

## 3. Where we deviated from the problem statement, and why

| Prescribed | What we did | Evidence |
| --- | --- | --- |
| Browsing track uses *dense retrieval* | Popularity prior + lexical matching | ProductAgent: dense collapses to 8.27 vs BM25 39.48 in conversational setting; our constraints are literal catalog substrings |
| Pipeline ends in *LLM Semantic Ranking* | Additive lexical ranking, no model | Network may be disabled at scoring; zero-token probe already reached 0.852 |
| Intent override performs *slot erasure* | Slot decay ×0.35 | All constraints derive from one target product; erasure discards valid evidence. Override MRR 0.9048, best of four scenarios |

Each deviation preserves the *intent* of the pillar and replaces the *mechanism*
on measured grounds. We would rather be asked to defend three documented
deviations than ship an unexamined literal implementation.

---

## 4. Using the read-only catalog as an arbiter

Two problems that present as NLP problems were solved by querying the catalog.

**Segmentation.** The simulator joins constraints with `"; "`, but constraint
strings contain internal semicolons of their own — `"solids: 100% cotton;
heathers: 75% cotton, 25% polyester"` is a single constraint. Naive splitting
fragments it. Resolution: if the joined span occurs **verbatim** in the candidate
pool, the semicolon is internal and the parts belong together; if not, the
simulator created the join. Extraction accuracy rose **94.0% → 97.5%**.

**Paraphrase robustness.** Extraction initially depended on the literal marker
`"is: "`; under paraphrase the chain went silent and buying-scenario Hit Rate
fell to 0.75. Extending a marker vocabulary would only overfit to the specific
paraphrase we invented. Instead `recover_spans()` ignores sentence structure and
recovers the longest spans appearing verbatim in the candidate pool — because
constraints *are* catalog text, they can be located without parsing the sentence
that carries them. Paraphrased score **0.7108 → 0.8135**, with the offline score
unchanged to six decimal places.

This required one enabling optimisation: `Catalog.bucket_blob()` concatenates a
bucket into a single cached string, turning each verbatim lookup from 184
Python-level iterations into one C-level substring search.

---

## 5. What we tried that did not work

| Approach | Result | Diagnosis |
| --- | ---: | --- |
| Information-gain clarification policy | 0.891142 → 0.891142 | 71.5% of sessions convert on turn 1; the policy rarely gets to speak |
| Constraint rarity as score multiplier | 0.891142 → **0.875861** | Annihilates signal when the target matches only common constraints; Hit Rate 1.00 → 0.98 |
| Constraint rarity as secondary sort key | 0.891142 → 0.891142 | With one constraint known, tied candidates share identical rarity |
| **Personalisation via `preference_tags`** | **rejected pre-implementation** | Measured first: rank-1 within ties 34/73 → 17/73, mean rank 5.53 → 29.30, worse in 49 of 73 |
| Field-weighted matching as tie-break | 0.891142 → **0.880908** | Buying MRR 0.694 → 0.674 |

Every regression was rolled back. The rule was fixed in advance: a change that
scores below the previous checkpoint is reverted, not retained pending
optimisation.

**The `preference_tags` row is the one we would highlight.** It was a scheduled
task, and it sounded obviously correct — personalise ranking using the supplied
profile. Tags like `fit`, `comfort`, `durability` turn out to appear in most
product copy and correlate with nothing about which specific item was purchased.
Measuring before building is the only reason it is not in the shipped system.

**The MRR diagnosis, for whoever attacks it next.** 143 of 200 sessions convert
on turn 1, but only 71 land at rank 1. In buying sessions the turn-1 tie group
has median size 36 and exceeds 10 in 65 of 80 sessions — the ten available slots
cannot hold the tie, so tie-breaking quality alone determines MRR. With a single
constraint known, every candidate containing it holds an identical score *and*
an identical rarity profile, which is why three separate tie-break formulations
changed nothing. The next thing we would try is a ranking prior conditioned on
the category bucket rather than the global popularity prior.

---

## 5b. The honest limit of what a 1.000 Hit Rate means

This section exists because a reviewer could reasonably level the following
charge at us, and they would be substantially right.

We declined the `"other"` shortcut on the grounds that it exploits an artifact of
the simulator rather than reflecting insight about shopping. But the system's
main mechanism rests on a closely related artifact. The simulated customer's
requirements are sliced **verbatim** out of the target product's own metadata, so
a phrase like `"Shaft measures approximately 8.37\" from arch"` is not something a
person would ever say — it is a fragment of a spec sheet. Our highest-weighted
signal is substring-matching exactly those fragments.

So: **Hit Rate@10 of 1.000 does not mean this agent is good at conversational
shopping. It means it is good at this simulator.** Against real customers, who
paraphrase, approximate, and describe products in words that appear nowhere in
the catalog, the verbatim route would contribute close to nothing, and
performance would fall back to what the category pruning, the IDF coverage
fallback, and the popularity prior can carry on their own. We have not measured
that number and we do not claim it.

What separates the two cases, in our view, is not that one is clean and the other
is not. It is that `"other"` bypasses the customer entirely — it extracts
requirements without modelling the person at all — whereas verbatim matching
still requires listening to what the customer said, tracking it across turns, and
deciding what to ask next. That is a difference of degree, not of kind, and we
would rather state it plainly than let the headline number stand unqualified.

The components we would expect to survive a transfer to real customers are the
architecture rather than the matcher: bucket pruning, additive scoring with no
conjunctive filtering, slot decay on intent override, and clarification chosen by
expected information gain. The verbatim fingerprint route is the part that is
fitted to this benchmark, and it is also the part carrying most of the score.

---

## 6. A property of the evaluator we found and chose not to exploit

`evaluator/local_evaluator.py`, customer reply logic:

```python
matches = [v for v in constraints
           if v not in disclosed
           and (attribute == "other" or classify_constraint(v) == attribute)][:2]
```

When `ask_attribute` is `"other"`, the type filter is bypassed and two
undisclosed constraints are returned regardless of type. Every session holds
exactly four constraints, so **asking `"other"` twice extracts the customer's
entire requirement set** without any understanding of the customer at all.

`"other"` is a permitted value in the API contract, and using it would raise our
score. We did not.

It is a property of the simulator's implementation, not an insight about
conversational commerce, and a system resting on it would not survive one day
against real shoppers. Disclosing it seemed more useful to the organisers than
quietly taking the points — and a submission that cannot distinguish a genuine
signal from an artifact of its own test harness has not understood its problem.

---

## 7. Reproducibility

No third-party packages. `requirements.txt` is empty by design.
Verified on Python 3.14.4 only; the 3.10 floor stated elsewhere is inferred from
the syntax used and has not been executed.

| Command | Verifies | Expected |
| --- | --- | ---: |
| `python -m tools.run_eval --gate 0.891` | Headline score | `0.891142` |
| `python -m tools.run_eval --paraphrase --gate 0.80` | Paraphrase robustness | `0.813505` |
| `python -m tools.data_health` | Catalog structure claims | 1115 buckets, 200/200 |
| `python -m tools.extract_audit` | Extraction accuracy | `97.5%` |
| `python -m tools.rule_parity` | Reimplemented rule == organiser's | 0 / 50,000 mismatches |
| `python -m tools.bench_selftest` | Benches are not fake green lights | dummy agent scores `0.0` |
| `python -m tools.failure_drill` | Fault handling has no side effects | 0 escapes, 0 empty returns |
| `python -m tools.demo_session` | One full multi-turn session | turn-by-turn trace |

**Two of these deserve emphasis.** `bench_selftest` feeds a deliberately broken
agent through both evaluation benches and requires the score to collapse to
0.0 — a bench that always reports green is more dangerous than no bench, because
it accelerates work in the wrong direction. `rule_parity` compares our
reimplemented category rule against the organiser's across all 50,000 products
on every run, because `src/` may not import `evaluator/` (the organiser's harness
will not contain our copy) and a silent divergence there would be invisible.

**Verified reproduction:** a fresh `git clone` into an empty directory with all
proxy environment variables cleared produces `0.891142`, identical to the
development environment.

---

## 8. Cost, latency, and feasibility

| | |
| --- | --- |
| Model API | None |
| Tokens (prompt + completion) | **0** |
| Monetary cost | **$0.00** |
| Network calls | **0** |
| Third-party dependencies | **0** |
| Index build | 0.85 s, one time |
| Per-session latency | **0.031 s** |
| Peak traced memory | **0.61 GB** (50,000-product index, measured with `tracemalloc`) |

The problem statement rules out heavy external vector-database clusters and
requires in-memory execution. This system satisfies that by construction rather
than by concession: there is nothing to deploy beyond a Python process and a
JSONL file.

## 9. Limitations

**MRR is unsolved and we say so.** 0.0976 of the remaining 0.109 points sit
there. Five approaches failed. §5 carries the full diagnosis so the next attempt
does not repeat ours.

**The paraphrase perturbation is our own.** The 0.8135 figure is measured against
a rewrite we authored. The private set may paraphrase differently, or not at
all. Span recovery is structure-independent by construction, which is the best
generalisation argument available to us — but it is an argument, not a
measurement.

**Extraction over-merges in 5 of 200 sessions**, where two short constraints
(`"imported; zipper closure"`) happen to appear adjacently somewhere in the
bucket. Mean recall remains 98.4%.

**One public-set constraint is in Chinese** (`进口`), which whitespace-delimited
span recovery misses. A non-whitespace tokenisation path would fix it.

**We specified three acceptance thresholds before measuring the metrics they
gated**, and had to correct all three. The development log records each. The
practice we adopted afterwards — measure a baseline, then set the gate — is the
single process change we would carry into the next project.

---

## 10. Data

Catalog and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD); see
`DATA_ATTRIBUTION.md`. The 50,000-product catalog was treated as strictly
read-only throughout: no mutation, no injected ASINs, no reconstruction of
upstream data. It is not committed to the repository.
