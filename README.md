# Offline Shopping Copilot — TechJam 2026 Track 4

A multi-turn conversational shopping agent that finds a customer's hidden target
product inside a frozen 50,000-item Amazon catalog, within at most 10 turns.

**TechnicalScore 0.891142** on the 200-session public set, against an official
weak-BM25 baseline of **0.10671** — an 8.35× improvement, using **zero LLM calls,
zero network access, and the Python standard library only**.

| Metric | Official baseline | This agent |
| --- | ---: | ---: |
| Hit Rate@10 | 0.125 | **1.000** |
| MRR | 0.068034 | **0.674806** |
| MTTC | 9.81 | **1.565** |
| Efficiency | 0.119 | **0.9435** |
| **TechnicalScore** | **0.10671** | **0.891142** |
| Tokens consumed | 0 | **0** |
| Wall clock, 200 sessions | — | **~6 s** (index build 0.85 s) |

> **The main deliverable document is [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md)**,
> in the root of this repository. It maps the system to the brief's four pillars,
> records every measurement including the failed experiments, and carries a Chinese
> summary alongside the English text. This README is the shorter orientation.
>
> **Demo video:** https://youtu.be/gyuh9YjiGA4 (2:39)

---

## Why it is offline by design

The official submission rules state that **for official final scoring, organizer
policy may disable network access**, and requires each team to declare whether
their system needs it. Pillar I of the brief prescribes a pipeline ending in LLM
semantic ranking; any such pipeline needs a declared offline fallback to survive
that condition. We have no visibility into what other entries did and make no
claim about them.

So the primary decision on this project was made before any code was written:
**the entire scoring path runs offline.** No model API, no embeddings, no vector
database, no third-party service. `requirements.txt` is empty on purpose.

This is not a limitation we worked around — it is the design. A probe built in
the first hour confirmed that the structure of the task, not model capacity, is
what the score responds to.

---

## Architecture

```
customer message
      │
 ┌────▼──────────────────────────────────────────────┐
 │ L0  Session state machine        src/session_state │  incremental slots,
 │                                                    │  intent override, budget
 ├────────────────────────────────────────────────────┤
 │ L1  Dual-track intent routing    agent.py          │  Buying → narrow
 │                                                    │  Browsing → popularity prior
 ├────────────────────────────────────────────────────┤
 │ L2  Multi-route retrieval        src/catalog       │  R1 category-bucket pruning
 │                                  src/ranker        │  R2 verbatim fingerprint match
 │                                                    │  R3 IDF-weighted coverage
 ├────────────────────────────────────────────────────┤
 │ L3  Ranking                      src/ranker        │  strictly additive, no hard
 │                                                    │  filters, popularity tie-break
 ├────────────────────────────────────────────────────┤
 │ L4  Clarification policy         src/askpolicy     │  live pool divergence ×
 │                                                    │  measured constraint priors
 └────────────────────────────────────────────────────┘
      │
 message · ask_attribute · 10 ranked parent_asin
```

### The three decisions that carry the score

**1. Category-bucket pruning is the only hard filter we allow.**
The customer's opening line contains a coarse category phrase. Rebuilding the
organiser's bucketing rule over the catalog yields 1,115 buckets with a median
of 184 items — a 271× reduction of the search space. We measured the target's
containment rate at **200/200**, which is what makes this safe. Every other
signal is additive.

**2. Constraints are verbatim catalog slices, so we match them as fingerprints.**
The simulated customer's stated requirements are sliced directly out of the
target product's own `features` / `details` text. A phrase like
`"Long torso camisole for extra coverage with spagetti adjustable strap"` is
effectively a product fingerprint. Substring matching it against the catalog
does more work than any semantic model would here.

**3. Nothing is ever hard-filtered out.**
Every constraint contributes an additive score; a mis-parsed constraint costs a
little ranking quality instead of eliminating the correct answer. This follows from ProductAgent's
report that LLM-generated SQL produced *trivial* queries in 55.36% of cases with
GPT-4 (their Table 7) - queries degenerate enough to stop discriminating. Note
this is "trivial", not "empty": we cite it as evidence that conjunctive filtering
degrades as constraints accumulate, not as a claim about empty result sets.

### The catalog as an arbiter

Two problems that look like NLP problems were solved by querying the read-only
catalog instead.

*Segmentation.* The simulator joins constraints with `"; "`, but constraint
strings contain internal semicolons of their own. Splitting naively fragments
one constraint into several. Resolution: if the joined span occurs **verbatim**
in the candidate pool, the semicolon is internal and the parts belong together;
if it does not, the simulator created the join. This lifted extraction accuracy
from 94.0% to **97.5%**.

*Paraphrase robustness.* Extraction initially depended on the literal marker
`"is: "`. Under paraphrased phrasing the whole chain went silent. Rather than
extending a list of marker phrases — which would only overfit to the specific
paraphrase we invented — `recover_spans()` ignores sentence structure entirely
and recovers the longest spans that appear verbatim in the candidate pool. The
paraphrase-perturbed score went from **0.7108 to 0.8135** with **no change to
the offline score**.

---

## Setup

Developed and verified on **Python 3.14.4**. No third-party packages.
The source uses no syntax newer than 3.10, but it has **only been executed on
3.14.4** - the 3.10 floor is inferred from the syntax used, not tested.

```bash
git clone https://github.com/LeyanPeng/Tiktok.git
cd Tiktok
```

**No organiser-supplied material is redistributed in this repository** — not the
catalog, not the public session set, not the evaluator. Copy them in from the
official participant kit:

```bash
# from an extracted copy of techjam-participant-kit.zip
cp -r <kit>/evaluator <kit>/starter <kit>/data .
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

That leaves `data/catalog.jsonl`, `data/public_set.jsonl`, `evaluator/`, and
`starter/` in place, which is everything the commands below need.

> Verify `catalog.jsonl.gz` against the kit's published `SHA256SUMS` before
> extracting. Note the checksum covers the `.gz` archive, not the extracted
> `catalog.jsonl` — checksumming the extracted file will not match, and that is
> expected rather than a sign of corruption.
>
> Attribution and terms of use for the underlying data: `DATA_ATTRIBUTION.md`.

## Reproducing every number in this README

```bash
python -m tools.run_eval --gate 0.891
```

Each of the following is a self-contained check that exits non-zero on failure.

| Command | Verifies | Expected |
| --- | --- | ---: |
| `python -m tools.run_eval --gate 0.891` | Headline score | `0.891142` |
| `python -m tools.run_eval --paraphrase --gate 0.80` | Paraphrase robustness | `0.813505` |
| `python -m tools.data_health` | Catalog structure claims | 1115 buckets, 200/200 |
| `python -m tools.extract_audit` | Constraint extraction accuracy | `97.5%` |
| `python -m tools.rule_parity` | Our category rule == organiser's | 0 mismatches / 50,000 |
| `python -m tools.bench_selftest` | The benches are not fake green lights | dummy agent scores `0.0` |
| `python -m tools.failure_drill` | Fault handling has no side effects | 0 escapes, 0 empty returns |
| `python -m tools.prior_audit` | Derived clarification prior matches the measured one | ordering identical |
| `python -m tools.constraint_count_stress` | Stopping rule survives a different constraint count | no regression |
| `python -m tools.demo_session` | One full multi-turn session, turn by turn | annotated trace |

**Network access is not required at any point**, for development or for scoring.
There is no online path to fall back from.

## Layout

```
TECHNICAL_REPORT.md         >> the main deliverable: full report, mapped to the
                               four pillars, with every measurement and every
                               failed experiment
README.md                   this file

agent.py                    submission entry point, exports Agent
src/catalog.py              read-only in-memory index, bucketing, priors
src/extract.py              category matching, constraint recovery, override detection
src/session_state.py        slot accumulation, override decay, turn budget
src/ranker.py               additive scoring and ranking
src/askpolicy.py            information-gain clarification policy
tools/                      evaluation harness and verification scripts
docs/*.json                 raw measured outputs backing every claim above
```

`src/` and `agent.py` deliberately **never import from `evaluator/`** — the
organiser's harness will not contain our copy. The category rule is therefore
reimplemented, and `tools/rule_parity.py` verifies the reimplementation against
the official one across all 50,000 products on every run.

---

## The honest limit of what 1.000 Hit Rate means

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

## What we tried that did not work

These are reported because negative results measured carefully are part of the
engineering record, and because two of them were planned features that would
have made the system worse.

| Approach | Result | Why |
| --- | ---: | --- |
| Information-gain clarification policy | 0.891142 → 0.891142 | 71.5% of sessions convert on turn 1; the policy never gets to speak |
| Constraint rarity as a score multiplier | 0.891142 → 0.875861 | Annihilates signal when the target only matches common constraints; Hit Rate fell 1.00 → 0.98 |
| Constraint rarity as a secondary sort key | 0.891142 → 0.891142 | With one constraint known, tied candidates share identical rarity |
| **Personalisation via `preference_tags`** | **rejected before implementation** | Measured first: rank-1 within ties fell 34/73 → 17/73, mean rank 5.53 → 29.30 |
| Field-weighted matching as a tie-break | 0.891142 → 0.880908 | Buying MRR fell 0.694 → 0.674 |

The `preference_tags` row is the one we would highlight. It was a scheduled task
in the plan and it sounded obviously correct — tags like `fit`, `comfort`,
`durability` are generic enough to appear in most product copy and correlate
with nothing about which specific item was purchased. Measuring before building
is the only reason it is not in the shipped system.

The clarification policy is retained despite showing no public-set gain: it
changes behaviour in the 28.5% of sessions that do not convert on turn 1, and it
is insurance for a private set with a lower turn-1 hit rate. We are not claiming
it helped here.

## A property of the evaluator we found and chose not to exploit

In `evaluator/local_evaluator.py`, the simulated customer's reply logic reads:

```python
matches = [v for v in constraints
           if v not in disclosed
           and (attribute == "other" or classify_constraint(v) == attribute)][:2]
```

When `ask_attribute` is `"other"`, the type filter is skipped entirely and two
undisclosed constraints are returned regardless of type. Since every session
holds exactly four constraints, **asking `"other"` twice extracts all of them**,
independent of any understanding of the customer.

It is permitted by the API contract, and it would raise the score. We did not
use it. It is a property of the simulator's implementation rather than an
insight about conversational commerce, and a system built on it would not
transfer to a real storefront for a single day. Reporting it seemed more useful
than quietly taking the points.

## Limitations, and what we would do next

**MRR is where the remaining score lives, and we did not crack it.** Hit Rate is
saturated at 1.000 and MTTC is near its structural floor (intent-override
sessions cannot convert before turn 3–4 by rule). Of the remaining 0.109 points,
MRR holds 0.0975. The diagnosis is precise: 143 of 200 sessions convert on turn
1, but only 71 of those land at rank 1. In buying sessions the turn-1 tie group
has a median size of 36 and exceeds 10 in 65 of 80 sessions — the top-10 slots
cannot hold the tie, so tie-breaking quality directly determines MRR. Five
approaches failed to improve it. A sixth idea we ran out of time for: learning a
ranking prior from the review-count and price distribution *conditioned on the
category bucket*, rather than the global popularity prior used now.

**The paraphrase perturbation is our own invention.** The 0.8135 robustness
figure is measured against a rewrite we wrote ourselves. The private set may
paraphrase differently or not at all. The span-recovery design is
structure-independent by construction, which is the best generalisation
argument we can make, but it is an argument rather than a measurement.

**Extraction still merges occasionally.** 5 of 200 sessions over-merge two short
constraints (`"imported; zipper closure"`) because the joined string happens to
appear verbatim somewhere in the bucket. Mean recall is 98.4%.

**One constraint string in the public set is in Chinese** (`进口`, "imported"),
which our span recovery misses. A tokenisation path that is not
whitespace-delimited would fix it.

## Data

Catalog and sessions derive from Amazon Reviews 2023 (McAuley Lab, UCSD). See
`DATA_ATTRIBUTION.md`. The catalog is treated as strictly read-only throughout.

## Team

Solo entry.
