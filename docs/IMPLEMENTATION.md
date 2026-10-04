# TASK-RL-200 / MS-RL-36 — implementation and verification

## MS-RL-36-L1 Plan

Create a separate local repository for the user-created pyjevsim-rl remote.
Follow [DESIGN.md](DESIGN.md) requirements and direct checks; do not repoint the
gorti repository, modify its staged changes, push, rerun old experiments, or
launch the new24-arm performance experiment as part of packaging.

## MS-RL-36-L1 Do

- Root: byte-preserved selected source import, preset/design, repository isolation,
  provenance, documentation and integration verification.
- Parallel implementation: runner/worker; bootstrap/dependency lock; analysis/host
  observations. Review feedback is resolved in this repository only.
- Imported150 source files, with original file size/hash and source HEAD recorded
  in `provenance/source-import.json`. This is a selected working-tree export,
  not a claim that all bytes are identical to the original HEAD commit.
- Native PyJevSim metadata2.1.1 is not substituted for source qualification.
  Existing engine provider's12 byte pins are unchanged. Git attributes prohibit
  newline translation of imported sources. dill0.4.1 universal-wheel hash is fixed.
- One entry point prepares an isolated repository-local venv, then invokes the
  actual new24-arm experiment and its analysis. Help/setup-only do not run models.
- No profiler, test suite, smoke run or freeze-state gate is in the experiment path.

## MS-RL-36-L1 Review — checkpoint

Windows CPython3.14.0: bootstrap setup-only succeeded; dedicated dill0.4.1 was
installed with its wheel hash. Help succeeded without setup; a second setup-only
call reused the environment without reinstalling. Focused46 tests passed:
analysis/host9, bootstrap17, design5, imported-source1, runner13, actual-model integration1.
The integration check uses4 tiny arms (Q/M × N/C1, L1/B1/suffix8), including
paths with spaces, and observed4/4 success and2/2 exact physical comparisons.
These timings are not performance evidence. Full24-arm experiment not executed.

The initial integration fixture accidentally requested an unsupported L2 action
plan; the kernel correctly rejected it. The fixture was corrected to declared L1
without changing any imported model or measurement source. No research arm was
retried or replaced. Cross-agent review corrected analyzer completion/fallback
field mismatches before final verification. Final-summary write failures preserve
terminal denominators on stderr and cannot be reported as study success. Owned
timeout/escalation and failed/mismatched/incomplete pairs have direct mock coverage.

Linux WSL Ubuntu24.04 CPython3.12.3 is available but its system Python lacks
ensurepip/python3-venv. Bootstrap reports manual distribution-package prerequisites;
it does not install system packages or overwrite partial environments. An isolated
Linux verification checkout is being prepared. Linux functional verification and
clone byte checks are not yet asserted in this checkpoint.

## MS-RL-36-L1 Reflect — checkpoint

The artifact isolates the portability/interference question, not a new performance
claim. C1 is the existing implementation, without changing its costly source
validation. Native N already uses snapshots. CPU endpoint remains distinct from
the preserved application-wall endpoint; host aggregates include benchmark work.
Six pairs per model support a small exploratory study, not confirmation or a
general claim. Fresh physical Linux hardware remains for the user's measurement.

Original repository check: HEAD4852206bdfcfa770df4d34fb1a0251fb87b811e1;
285 staged raw entries, joined-LF excluding-final-newline SHA256
`1acb8d739725ff0ca189cac0ad8b658e640cfc1727ad81db889ccf6d2f13085c` unchanged;
original origin remains cbchoi/gorti. New origin is cbchoi/pyjevsim-rl,
branch `codex/portable-benchmark`. No remote push.

Same-purpose corrective task additions0; this is one implementation/review cycle.
Source/history publication remains withheld pending a separate user instruction.
