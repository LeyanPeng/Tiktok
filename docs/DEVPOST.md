# Devpost Submission — paste-ready

> Copy the sections below into the Devpost project description.
> Everything between the rules is submission text; the notes in blockquotes are
> for you and should not be pasted.

---

## Offline Shopping Copilot

**A multi-turn conversational shopping agent that finds a customer's hidden
target product inside a 50,000-item Amazon catalog — scoring 8.35× the official
baseline with zero LLM calls and zero network access.**

### The result

| Metric | Official baseline | This system |
| --- | ---: | ---: |
| Hit Rate@10 | 0.125 | **1.000** |
| MRR | 0.068034 | **0.674806** |
| MTTC (turns to conversion) | 9.81 | **1.565** |
| **TechnicalScore** | **0.10671** | **0.891142** |
| Tokens consumed | 0 | **0** |
| Cost per 200 sessions | — | **$0.00** |
| Latency per session | — | **0.031 s** |

### How it addresses the problem statement

The brief asks for four things. Here is what each became, and where we
deliberately departed from the literal prescription with evidence.

**I. Intent routing and a hybrid retrieval pipeline.** The opening message is
classified into a *Buying* track (a hard constraint is already present — rank
aggressively, ask sparingly) or a *Browsing* track (vague opener — fall back to
a popularity prior and spend a turn learning). Retrieval runs three routes
combined additively: category-bucket pruning that collapses 50,000 products to a
median of 184, verbatim fingerprint matching on the customer's stated phrases,
and IDF-weighted token coverage as a graded fallback.

**II. A dynamic dialog state machine.** Constraints accumulate as weighted slots.
When the customer changes their mind mid-session, earlier slots are *decayed*
rather than erased — in this task every constraint originates from the same
target product, so erasing them discards evidence still pointing at the right
answer. Intent-override sessions score MRR 0.9048, our strongest scenario. The
agent stops asking once it holds four constraints, which we measured to be
exactly how many every session contains; questions beyond that impose
conversational cost with no possible return.

**III. Self-evolution through runtime adaptation.** Clarification priorities are
recomputed every turn from the *live candidate pool*: expected value equals the
measured probability that a customer holds a constraint of that type, times the
entropy of that attribute across the surviving candidates. An attribute every
candidate agrees on carries zero information and asking about it burns a turn.
When the agent's own top-5 stops moving for two consecutive turns after
information has arrived, it switches selection criteria at runtime — dropping the
prior term and choosing purely by pool divergence.

**IV. The evaluation matrix, treated as a budget.** Once Hit Rate@10 saturated at
1.000, 89% of all remaining points sat in MRR. Every subsequent decision was
tested against that single question, and we report honestly that five separate
attempts failed to move it.

### The decision the brief made for us, that most entries will miss

The submission rules state that **for official final scoring, organizer policy
may disable network access**. Pillar I prescribes a pipeline ending in *LLM
Semantic Ranking* — a design that fails or degrades sharply under that
condition.

So we tested the premise before committing. In the first hour we built a
throwaway 130-line probe using only the Python standard library. It scored
**0.852** against a baseline of **0.107**. That settled it: the entire scoring
path runs offline, with no online path to fall back from. What the score
responds to here is the structure of the problem, not model capacity.

### Something we found and chose not to use

Reading the evaluator, we found that passing `"other"` as the clarification
attribute bypasses the simulator's type filter entirely and returns two
undisclosed constraints regardless of type. Since every session holds exactly
four constraints, asking `"other"` twice extracts the customer's entire
requirement set — with no understanding of the customer at all.

It is permitted by the API contract and it would raise our score. We did not use
it. It is a property of the test harness rather than an insight about
conversational commerce, and a system resting on it would not survive one day
against real shoppers. We would rather report it than quietly take the points.

### What we tried that did not work

| Approach | Result |
| --- | ---: |
| Information-gain clarification policy | no change — 71.5% of sessions convert on turn 1 |
| Constraint rarity as a score multiplier | 0.891 → 0.876 |
| Constraint rarity as a tie-break | no change |
| **Personalisation via user preference tags** | **measured first, rejected before building** |
| Field-weighted matching as a tie-break | 0.891 → 0.881 |

The personalisation row is the one worth dwelling on. It was a scheduled feature
and it sounded obviously right. Measured first: rank-1 placement within tied
candidates fell from 34/73 to 17/73, and mean rank went from 5.53 to 29.30. Tags
like *fit*, *comfort*, *durability* appear in most product copy and correlate
with nothing about which item was purchased. Measuring before building is the
only reason it is not in the shipped system.

### Verification

Every claim above is produced by a command in the repository that exits non-zero
on failure — including two checks aimed at our own instruments. `bench_selftest`
feeds a deliberately broken agent through both evaluation benches and requires
the score to collapse to 0.0, because a bench that always reports green is worse
than no bench. `rule_parity` compares our reimplemented category rule against the
organiser's across all 50,000 products on every run, because a silent divergence
there would be invisible.

A fresh `git clone` into an empty directory with all proxy environment variables
cleared reproduces `0.891142` exactly.

---

## Built with

**Development tools:** VS Code, Claude Code, Git, Windows 11 terminal (PowerShell
and Git Bash), Python 3.14.4.

**APIs used:** None. No model API, no external service, no network call at any
point in development or scoring.

**Libraries and frameworks:** Python standard library only — `json`, `re`,
`math`, `collections`, `dataclasses`, `pathlib`, `statistics`, `argparse`.
`requirements.txt` is empty by design. No PyTorch, no Transformers, no
scikit-learn, no vector database, no embedding model.

**Datasets and assets:** The official TechJam participant kit — a frozen
50,000-product catalog from the `Clothing_Shoes_and_Jewelry` category of the
Amazon Reviews 2023 dataset (McAuley Lab, UCSD), 200 labeled public development
sessions, and the deterministic local evaluator. The catalog was verified against
the published SHA256 checksum and treated as strictly read-only. No external data
was introduced.

**Prior work referenced:** ProductAgent (Jin et al., *Benchmarking Conversational
Product Search Agent with Asking Clarification Questions*, EMNLP 2025 Industry
Track, arXiv:2407.00942) informed three design choices — grounding clarification
in live candidate-pool statistics, avoiding conjunctive hard filtering, and
expecting lexical retrieval to beat dense retrieval in the conversational regime.

---

## Links

- **Repository:** https://github.com/LeyanPeng/Tiktok
- **Demo video:** _[paste YouTube URL here — must be set to Public]_
- **Technical report:** `docs/TECHNICAL_REPORT.md`
- **Development log, including every failed experiment:** `PROGRESS.md`

## Team

Solo entry.

---

> **Before you submit, check these:**
> - [ ] YouTube video visibility is **Public**, not Unlisted
> - [ ] Repository is **Public** — open it in a private/incognito window to confirm
> - [ ] Demo video URL is pasted into the description above **and** into Devpost's video field
> - [ ] Track 4 is selected on the submission form
> - [ ] No API keys anywhere in the repo (`git grep -i "api_key\|secret\|token"` — already verified clean)
