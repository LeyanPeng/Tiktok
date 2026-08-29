# Demo Video Script — 2 minutes 40 seconds

Official guidance for backend tracks: *"If a front-end interface is not
applicable, a walkthrough video showing API usage, inference examples, or result
analysis is accepted."* So this is a terminal walkthrough. No UI required, and
building one would have cost time the brief explicitly places out of scope.

**Before recording**

```bash
cd C:\Users\22300\techjam-track4
```

Set the terminal to a large font (16pt+) and a dark theme — recorded terminals
are unreadable at default size. Widen the window to at least 100 columns so the
tables do not wrap. Do a 20-second test recording first and play it back at the
size a judge will actually watch it.

**Requirements to respect:** YouTube, visibility **Public**, no third-party
trademarks or copyrighted content (no music, no brand logos, no stock footage).

---

## Shot 1 — The problem and the number (0:00–0:25)

**On screen:** `README.md` open, scrolled to the results table.

> "The task is to find a customer's hidden target product inside a fifty-thousand
> item catalog, in at most ten conversational turns. The official baseline scores
> point-one-oh-seven. This system scores point-eight-nine-one — eight-and-a-third
> times higher.
>
> It does that with zero LLM calls, zero network access, and no third-party
> packages. That was not a constraint I worked around. It was the first design
> decision, and it came from reading the submission rules."

---

## Shot 2 — Why offline (0:25–0:50)

**On screen:** `docs/submission_rules.md`, highlight the line about network
access being disabled at final scoring.

> "The rules say the organisers may disable network access for final scoring. The
> problem statement asks for a pipeline ending in LLM semantic ranking — a design
> that fails outright under that condition.
>
> So before committing to an architecture, I tested the premise. A hundred-and-
> thirty-line probe, standard library only, first hour of the project: it scored
> point-eight-five-two. What this task responds to is the structure of the
> problem, not model capacity."

---

## Shot 3 — A full session, live (0:50–1:50) — **the core of the video**

**Run this:**

```bash
python -m tools.demo_session --scenario intent_override
```

Let the output render, then narrate over it. Pause on each turn.

> "Here is one complete session. The customer wants a belt — and mid-conversation
> they will change their mind, which is the hardest of the four scenarios.
>
> **Turn one.** The opening line contains a category phrase. That single phrase
> prunes fifty thousand products down to two hundred fifty-eight — a two-hundred-
> seventy-fold reduction. I measured the target's containment rate at two hundred
> out of two hundred, which is what makes it safe to use as a hard filter. It is
> the only hard filter in the system. The agent extracts one constraint, and the
> target is already sitting at rank two.
>
> **Turn two.** The agent asks about *feature* — not from a fixed list, but
> because it recomputed, against the surviving candidates, which attribute would
> split them most. One more constraint arrives, and the target moves to rank one.
>
> **Turn three.** The customer changes their mind. The agent detects it and
> *decays* the earlier constraints rather than erasing them — because in this
> task every constraint comes from the same target product, so erasing them
> throws away evidence that still points at the right answer. Target holds rank
> one. Converted on turn three, first place, zero tokens."

---

## Shot 4 — What did not work (1:50–2:20)

**On screen:** the failure table in `docs/TECHNICAL_REPORT.md` §5.

> "Five things I tried that did not work, and I am showing them because the
> negative results were more informative than the wins.
>
> The one worth your attention is personalisation. Using the supplied customer
> preference tags to rank was a scheduled feature and it sounded obviously
> correct. I measured it before building it: rank-one placement among tied
> candidates fell from thirty-four out of seventy-three, to seventeen. Mean rank
> went from five-point-five to twenty-nine. Tags like *fit* and *comfort* appear
> in most product copy and correlate with nothing about which item was bought.
>
> Measuring before building is the only reason that is not in the shipped
> system."

---

## Shot 5 — The thing I found and did not use (2:20–2:40)

**On screen:** `docs/TECHNICAL_REPORT.md` §6, the `matches = [...]` snippet.

> "Last thing. Reading the evaluator, I found that passing `other` as the
> clarification attribute skips the type filter entirely and hands back two
> undisclosed constraints. Every session holds exactly four. So asking `other`
> twice extracts the customer's entire requirement set — with no understanding of
> the customer at all.
>
> It is allowed by the contract and it would raise my score. I did not use it.
> It is a property of the test harness, not an insight about shopping, and a
> system built on it would not survive one day against real customers.
>
> Everything I have shown reproduces from a clean clone with the network off.
> Thank you."

---

## Recording checklist

- [ ] Terminal font ≥ 16pt, ≥ 100 columns, dark theme
- [ ] 20-second test recording reviewed at final playback size
- [ ] `python -m tools.demo_session --scenario intent_override` runs clean beforehand
- [ ] Close Slack, mail, notifications — nothing pops into frame
- [ ] No music, no logos, no stock footage
- [ ] Total runtime under 3 minutes
- [ ] Uploaded to YouTube with visibility **Public** (not Unlisted)
- [ ] URL pasted into the Devpost description **and** Devpost's video field
- [ ] Opened the final YouTube link in a private window to confirm it plays

## If narration is difficult

Official guidance accepts a walkthrough without voice-over. Record the terminal
session and add on-screen captions carrying the same lines. Content matters more
than delivery — the judging criterion is whether the work is communicated
clearly, not whether the accent is polished.

## Backup shots, if time allows

```bash
python -m tools.demo_session --scenario browsing --failure   # an honest failure case
python -m tools.bench_selftest                               # proving the bench is not a fake green light
python -m tools.run_eval --paraphrase --gate 0.80            # robustness under paraphrase
```

The failure case is worth including if the video has room. Showing a case the
system gets wrong, with a diagnosis, demonstrates more understanding than any
additional success would.
