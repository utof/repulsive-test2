# Branch `feat/phase2h-impl` — whole-branch review context

- Merge base with `main`: `9dd5551`. 47 commits.
- Diff of `src/` + `test/` only: `docs/superpowers/plans/phase2h-review-ledger/branch-src-test.diff` (28 199 lines, 43 files).
- The contract is the design spec: `docs/superpowers/specs/2026-10-09-phase2h-bct-gmres-design.md` (~5 000 lines).
  **It carries 47 numbered amendments (A1–A47) at the end, and a later amendment OVERRIDES the body
  wherever they conflict.** §9 is the ruling table; §7 holds the gates H1–H6/R1; §11.2 is the task table.
  Check amendments before reporting a body statement as violated — many body statements are superseded.
- Full command set: `bun test` (suite), `bunx tsc --noEmit` (typecheck), `bunx biome check` (lint).
  **CORRECTED 2026-10-12:** an earlier draft of this file said `lefthook.yml` runs biome with
  `--linter-enabled=false` and therefore never lints. That was true until **this branch's commit
  `f037cf1` removed the flag**, so the hook DOES lint now (`lefthook.yml:18`). Verify lint yourself
  regardless — do not cite a hook's exit code as evidence either way.
- Run every command in the FOREGROUND. Do not background anything; do not spawn subagents.
