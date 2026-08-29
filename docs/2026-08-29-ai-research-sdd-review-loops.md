# AI research: SDD review loops — round caps, blind vs informed re-review, severity gates

Started 2026-08-29. Basis for the "Review loop" section of `CLAUDE.md`.

## A. obra/superpowers (Jesse Vincent)

### subagent-driven-development/SKILL.md (fetched 2026-08-29, main branch)
- **5 rounds max per task.** Rounds 1-3 resume the ORIGINAL implementer; rounds 4-5 dispatch a FRESH implementer on a more capable model.
- **Loop terminates, does not run until approval.** "When round 5's re-review still leaves findings open, stop dispatching. Adjudicate each open finding yourself" — the controller/orchestrator adjudicates.
- **Severity gate:** "Record Minor findings in the progress ledger... Minor findings never enter the loop." Critical + Important are blocking.
- **Scoped re-review:** after each fix round the re-review is scoped to the changed range via `scripts/review-package PLAN_FILE FIX_BASE HEAD`, not a full re-review of the task.
- **Spec + quality both required, one report:** "never accept a report missing either verdict — spec compliance AND task quality are both required."

### requesting-code-review/SKILL.md (fetched 2026-08-29)
- Severity tiers: Critical = fix immediately; Important = fix before proceeding; Minor = note for later.
- Fresh context per dispatch: "Hand it precisely crafted context, never your session's history. That keeps the reviewer on the work product, not your thought process."
- Does NOT define multi-round / re-review procedure or a formal approval artifact. (Gap; the round machinery lives in subagent-driven-development.)

### re-review-prompt.md (fetched 2026-08-29) — THE key artifact for "fresh reviewer can't approve"
Superpowers solves "a fresh reviewer can't approve because it never saw the prior bar" by making the re-reviewer **informed and narrowly scoped**, not fresh:

> "Use this template when dispatching a re-review after a fix round... **It is not a fresh review — the full review already happened.**"
> "**Purpose:** Verify each finding from the previous review was addressed, and that the fix itself broke nothing."
> "Your scope is the findings list and the fix diff. Verdict every finding... Do NOT re-review code the fix did not touch: if you notice an issue entirely outside the fix diff, report it under Out-of-Scope Observations — **it does not block this task and does not extend the loop.**"

Output contract per finding: `ADDRESSED | NOT ADDRESSED` with file:line. "'Attempted' is not addressed: the specific defect must no longer exist." Round verdict is binary: "All findings addressed, no new Critical/Important breakage | Findings remain open."

Anti-cost rules in the same prompt: "You Do Not Dispatch Subagents... a reviewer you spawn duplicates one of them at full cost, and its verdict counts for nothing." And: don't re-run suites, trust-but-verify the implementer's test claims against the diff; run only focused tests for specific doubts.

### The escalation ladder + breaker (verbatim, SKILL.md)
- "**Rounds 1-3 — resume the original implementer.** Send it the open findings verbatim. Its context is intact: it knows the task, the code, and its own choices."
- "**Rounds 4-5 — dispatch a fresh implementer on a more capable model**... 'A prior implementer attempted this task [N] times; you own it now.' A loop that survives three resumes usually means the implementer cannot see its own problem — fresh eyes and a capability bump in one move."
- "**The breaker.** When round 5's re-review still leaves findings open, stop dispatching. Adjudicate each open finding yourself — you hold the plan and the cross-task context the reviewer lacks." Three adjudication outcomes: reviewer wrong/contestable → park with ruling; real but nothing downstream builds on it → park with ruling; real AND load-bearing → "rule on the smallest change that unblocks the dependent work" and carry into the next task's dispatch.
- "**Adjudicate only at the cap. Adjudicating earlier to end a loop is pre-judging with a different name.** Every adjudication is a ledger entry — a silent discard is forbidden."
- Anti-pre-judging rule for the ORCHESTRATOR: "Do not pre-judge findings for the reviewer — never instruct a reviewer to ignore or not flag a specific issue... If the prompt you are writing contains 'do not flag,' 'don't treat X as a defect,' 'at most Minor,' or 'the plan chose' — stop: you are pre-judging, usually to spare yourself a review loop."
- "Never fix findings yourself in the controller session — your context stays clean for coordination, and controller fixes skip review."

### Final whole-branch review (after all tasks)
"If the final whole-branch review returns findings, dispatch ONE fix subagent with the complete findings list — not one fixer per finding. Per-finding fixers each rebuild context and re-run suites; **a real session's final-review fix wave cost more than all its tasks combined.** Then run exactly one scoped re-review of the fix wave... **There is no second fix wave** — residual load-bearing findings surface to your human partner."
⇒ Cap at the branch level is 1 round. Cap at the task level is 5.

### task-reviewer-prompt.md (fetched 2026-08-29) — spec+quality merged into ONE reviewer
Verdict vocabulary: Spec Compliance = "✅ Spec compliant | ❌ Issues found | ⚠️ Cannot verify from diff"; Task Quality = "Approved | Needs fixes". Both verdicts required in one report.
Severity: Critical = incorrect/fragile behavior, missed requirements, maintainability damage blocking merge; Important = cannot trust the task until fixed; Minor = coverage gaps, polish, non-blocking.
Anti-rubber-stamp: "Do not trust the report" (implementer claims are unverified); verify batched file lists line-by-line; missing files = findings.
Anti-nitpick: acknowledge strengths first; categorize by ACTUAL severity, not everything Critical; "coverage could be broader"/polish = Minor only; don't flag pre-existing file sizes.
Cost control: read the diff once, don't crawl the codebase beyond "one focused check per named risk", don't re-run full suites.

### RELEASE-NOTES.md — the empirical rationale (v6.0.0, 2026-06-16; latest v6.3.0, 2026-08-12)
Superpowers 6.0 was an explicit reversal of the two-reviewer-per-task design, driven by evals:

> "Superpowers 6.0 is a big release. The headline is a rewrite of how `subagent-driven-development` reviews each task — cheaper, stricter, and harder to game."
> "While these numbers won't hold on every harness and for every workload, in our evals, Claude Code and Codex produce similar high-quality results **roughly twice as fast and while spending almost 50% fewer tokens**."
> "A long run of cost-and-quality experiments on real projects reshaped how the controller reviews each task. **The old flow ran two reviewers per task** and leaned on the controller's judgment for model choice and severity, and **both turned out to be expensive and easy to game.**"

Specific changes and the failure each fixes:
- **"One reviewer per task, two verdicts."** `spec-reviewer-prompt.md` + `code-quality-reviewer-prompt.md` were DELETED and merged into `task-reviewer-prompt.md`. Rationale: "reads the task's diff once and returns both a spec-compliance verdict and a quality verdict, **so one fix pass clears both**." Plus a new "can't verify from the diff" verdict for requirements living in untouched code. (#1538, #1543)
- **"One broad review at the end"** on the most capable model, "instead of re-reviewing everything task by task."
- **Anti-rubber-stamp / anti-coaching:** "The controller can't tell a reviewer what to ignore. **Real runs caught controllers coaching reviewers to skip a finding or call it 'Minor at most,' and the flaw shipped.**" Suppressing findings and pre-rating severity are now banned outright.
- **Anti-rationalization:** "an implementer's 'I left this unabstracted on purpose' no longer talks a reviewer out of a real finding." Reviewers are read-only (a reviewer running `git checkout` had orphaned commits).
- **Cost:** diffs move as FILES not pasted text — "A pasted diff parks itself permanently in the most expensive context, and a reviewer without one rebuilds it by hand — **the single biggest reviewer cost.**"
- **Model drift:** "Left to choose, controllers stopped naming a model at all — and an unnamed model quietly inherits the session's most expensive one, so **one run put all 26 of its reviewers on the top tier.**"
- **ROUND-COUNT DATA POINT (writing-plans):** right-sizing tasks so each "earns its own test cycle and a reviewer's pass" — "In testing, **a plan written this way needed one round of fixes where the control needed two to four** — and the control shipped a real bug." ⇒ Round count is a function of PLAN QUALITY, not reviewer strictness. Fix the plan, not the loop.

v6.2.0-era notes (≤ 2026-08-12):
- "**Implementers and reviewers may not spawn their own subagents, which was producing duplicate reviews.**" (#2059)
- "Reviewers re-read evidence they find illegible instead of re-running the test suite (#2089)"
- "Controllers no longer stall on plan conflicts. Non-catastrophic conflicts and ambiguities get a recorded ruling and work continues... **One donated session had sat blocked for almost nine hours on a question the controller could have decided.**" (#2077)
- "Small same-shape tasks batch into one dispatch, cutting subagent cost sharply on micro-task plans." (#2078)
- v6.0.0 removed the SDD workspace contamination problem; ledger is plan-scoped `.superpowers/sdd/<plan-basename>/`.

**Direct relevance to a repo rule mandating TWO separate reviewers (spec + quality):** upstream tested that exact design and abandoned it as "expensive and easy to game," replacing it with one reviewer emitting two verdicts. The stated benefit of merging was that one fix pass clears both classes of finding.

## B. Other frameworks

### GitHub Spec Kit (github/spec-kit), `templates/commands/analyze.md` (fetched 2026-08-29, main)
Spec Kit's review is **pre-implementation artifact review, human-gated, and explicitly NOT a loop.**
- "**STRICTLY READ-ONLY**: Do not modify any files. Output a structured analysis report. Offer an optional remediation plan (**user must explicitly approve** before any follow-up editing commands would be invoked)."
- 4-level severity heuristic: CRITICAL (violates constitution MUST / missing core artifact / requirement with zero coverage blocking baseline) | HIGH (duplicate or conflicting requirement, ambiguous security/perf attribute, untestable acceptance criterion) | MEDIUM (terminology drift, underspecified edge case) | LOW (style/wording).
- **Severity gate on proceeding, not on re-review:** "If CRITICAL issues exist: Recommend resolving before [implement]. If only LOW/MEDIUM: **User may proceed**, but provide improvement suggestions."
- "Ask the user: 'Would you like me to suggest concrete remediation edits for the top N issues?' (**Do NOT apply them automatically.**)"
- Constitution conflicts are automatically CRITICAL and cannot be resolved by "dilution, reinterpretation, or silent ignoring" — a fixed non-negotiable authority beats reviewer judgement.
- `checklist.md`: checklists are "**UNIT TESTS FOR ENGLISH**" — requirements-quality validation, explicitly NOT implementation verification. "This command generates or appends checklist items; **it MUST NOT mark generated items `[x]`**" — the agent may not self-certify; the reviewer owns the checkbox.
- `implement.md`: "Treat checklist markers as a **read-only gate**: scan checkbox state, report status, and ask before proceeding when needed; do NOT modify checklist files or markers."
⇒ Spec Kit's anti-rubber-stamp mechanism is **structural separation of who may mark done**, not loop iteration. No round cap because there is no automated round.

## C. Empirical / academic (2025+). arXiv IDs verified by fetching arxiv.org/abs/<id> (citation_title + citation_date match).

### 1. "Iterative Audit Convergence in LLM-Managed Multi-Agent Systems: A Case Study in Prompt-Engineering Quality Assurance" — arXiv:2605.12280, 2026-05-12. Strongest direct evidence on round counts.
A production system (AEGIS) with a 7152-line specification surface was "audited across **nine rounds**, surfacing **51 consistency defects**."
- **Defects found per round: 15, 8, 12, 2, 8, 1, 4, 1, 0.**
- The paper's own term is **non-monotonic convergence**: rounds 3, 5 and 7 each found MORE than the round before. Attributed to "cascading edits and audit-scope expansion."
- "Single-file review missed defect classes that were surfaced only by later expanded-scope rounds."
- Stopping rule is empirical: terminated when a full cycle surfaced zero defects (round 9).
**Reading it honestly:** it took 9 rounds to reach zero, so a 2–3 round cap would have stopped mid-discovery. BUT the late spikes came from *scope expansion* and *cascading edits from the fixes themselves* — exactly what a **scoped** re-review prevents. N=1 case study on a spec/prompt surface, not application code.

### 2. "Refute-or-Promote: An Adversarial Stage-Gated Multi-Agent Review Methodology for High-Precision LLM-Assisted Defect Discovery" — arXiv:2604.19049, 2026-04-21. Strongest evidence on reviewer false positives and the limits of adding reviewers.
- "LLM-assisted defect discovery has a **precision crisis**: plausible-but-wrong reports overwhelm maintainers and degrade credibility." (HackerOne paused the Internet Bug Bounty in March 2026 over AI-amplified volume.)
- Design: adversarial agents **attempt to disprove** each candidate finding at every promotion gate.
- Measured: over 31 days "the pipeline **killed roughly 79% of 171 candidates**"; prospective subset "kill rate was **83%**."
- **Most important result:** "**ten dedicated reviewers unanimously endorsed a non-existent Bleichenbacher padding oracle in OpenSSL's CMS module; it was killed only by a single empirical test.**" ⇒ Reviewer ensembles and extra rounds do NOT correct a confident-but-wrong finding; an **executable check** falsified it.
- "cold-start reviewers are intended to reduce anchoring cascades" — the case FOR blind reviewers — but cold-start independence is not sufficient alone.

### 3. 2026 benchmarks establishing that long iterative agent loops DEGRADE (titles/dates verified, full results not fetched):
- "**SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks**" — arXiv:2603.24755, 2026-03-25.
- "**EvoCode-Bench: Evaluating Coding Agents in Multi-Turn Iterative Interactions**" — arXiv:2605.24110, 2026-05-22.

### Anthropic — official Claude Code docs, https://code.claude.com/docs/en/code-review (fetched 2026-08-29; "July 2026 update"). PRIMARY SOURCE.
**No approval verdict at all.** "Findings are tagged by severity and **don't approve or block your PR**." "The check run always completes with a **neutral conclusion**." ⇒ findings, not a verdict; no "approved" state to loop until.

**Verification-gating instead of round-looping.** "multiple agents analyze the diff... in parallel... then **a verification step checks candidates against actual code behavior to filter out false positives**."

**Severity taxonomy:** 🔴 Important (bug to fix before merging) | 🟡 Nit | 🟣 Pre-existing. "By default, Code Review focuses on **correctness: bugs that would break production, not formatting preferences or missing test coverage.**"

**Direct answer to "reviewers always find new issues"** — `REVIEW.md` tuning:
> "**Re-review convergence**: ... A rule like '**after the first review, suppress new nits and post Important findings only**' **stops a one-line fix from reaching round seven on style alone.**"
The prescribed fix for round inflation is a **severity ratchet on re-review**, not a round cap and not a fresh reviewer.

**Anti-false-positive knob:** "**Verification bar**: require evidence before a class of finding is posted. For example, 'behavior claims need a `file:line` citation in the source, not an inference from naming'."
**Nit volume cap:** "Report at most five Nits per review."
**Coverage/confidence is an explicit dial** (effort level). **Feedback does not re-trigger** a re-review; only push or explicit `@claude review` does. **Cost:** "$15-25 per review."

### 4. "Review Arcade: On the Human Alignment and Gameability of LLM Reviews" — arXiv:2605.28897, 2026-05-27. Evidence against "loop until the reviewer approves."
Domain: scientific peer review (analogical). "**limited alignment of LLM reviews with human ones.**" **Gaming result:** a "draft-revise workflow to improve the submission according to the LLM review" achieved **statistically significant score increases for ~35% of papers.** ⇒ Iterating against an LLM reviewer raises its score without the underlying quality necessarily moving (Goodhart).

### 5. "LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit" — arXiv:2604.19117, 2026-04-21 (title+date verified; abstract not fetched in full). Mechanistic support for the rubber-stamp risk: a reviewer handed the implementer's confident rationale may agree against its own assessment.

### 6. UNVERIFIED — do not cite
- Vendor comparisons (Greptile/CodeRabbit false-positive numbers, "CR-Bench") — SEO/affiliate sources only; discarded.
- BMAD-METHOD gate PASS / CONCERNS / FAIL / WAIVED — secondary sources only (primary repo paths 404'd). Design point: 3 of 4 verdicts let work proceed; WAIVED = justified escape hatch.
- Tessl, Cursor/Windsurf: no primary source located.

### OpenAI — "Custom Code Review rules for Codex" (developers.openai.com, undated, live 2026-08-29)
- Repository rules raised the **recovery rate of intended violations to 98% vs 58.3%** baseline. Warns broad instructions "create noise". ⇒ The one lever with a measured ~1.7× effect is REVIEWER CRITERIA QUALITY, not round count.

## SYNTHESIS TABLE
| Source | Rounds cap | Blind vs informed re-review | Severity gate | Spec/quality split |
|---|---|---|---|---|
| superpowers SDD v6.3 | **5/task**, then adjudicate; **1** fix wave at branch level | **Informed + scoped**; per-finding ADDRESSED/NOT ADDRESSED | Minor never enters the loop | **Merged**: one reviewer, two verdicts |
| Anthropic Code Review | **No loop**; re-review on push only | Informed; recommends a **ratchet** | Important/Nit/Pre-existing; nit caps | Correctness-only by default |
| Spec Kit `/analyze` | **No loop**; human approves remediation | N/A | CRITICAL blocks; LOW/MEDIUM proceed | Artifact consistency; agent may not self-certify |
| BMAD (secondary) | Advisory gate | N/A | PASS/CONCERNS/FAIL/WAIVED | Separate advisory QA agent |
| arXiv 2605.12280 | Ran 9; non-monotonic (15,8,12,2,8,1,4,1,0) | — | — | — |
| arXiv 2604.19049 | Stage gates | Cold-start reviewers, insufficient alone | Adversarial refutation kills 79–83% | — |
| arXiv 2605.28897 | — | — | — | Loop is gameable (~35%) |

## RECOMMENDATION (evidence-weighted) — applied in `CLAUDE.md` "Review loop"
1. Review 1 blind + full-scope; one reviewer, two verdicts; split only for numerics-critical work.
2. Every re-review informed + scoped: findings verbatim + fix diff; ADDRESSED/NOT ADDRESSED per finding; new findings count only if Critical/Important AND inside the fix diff.
3. Severity gate: Critical/Important only; from round 2 on, Important-and-above only.
4. Cap 3 rounds; escalate at round 3 (fresh implementer, one tier up); orchestrator adjudicates in writing at the cap; never earlier; no silent discard.
5. Critical findings must be falsifiable (file:line + failing test/executable check where feasible).
6. Orchestrator never tells a reviewer what to ignore or pre-rates severity.
7. Branch level: one blind whole-branch review → one consolidated fix → one scoped re-review → residuals to the human.

## WHERE THE EVIDENCE IS STRONG VS THIN
STRONG: severity-gated re-review (three primary sources converge); scoped re-review (superpowers + 2605.12280 mechanism); LLM review false-positive rate (79–83% measured); ensembles don't self-correct a confident wrong finding (OpenSSL existence proof); criteria quality beats round count (98% vs 58.3%).
THIN: the exact cap number (no published ablation; 3 is inferred from superpowers' 1-vs-2–4 observation — treat as tunable, log actual round counts); merged vs split reviewer for correctness-critical code (no public evidence); gameability transfer from peer review to code; blind vs informed FIRST review (indirect only).
