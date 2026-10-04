# Portable local continuation benchmark — TASK-RL-200 / MS-RL-36

## Plan / SRS (requirements first)

The user created `https://github.com/cbchoi/pyjevsim-rl.git` and requested remote
configuration plus a platform-independent experiment script. This is a separate
repository, not a replacement remote for gorti. No remote push or publication is
authorized in this task. The requested artifact is a runnable benchmark, not a
new performance conclusion or a federation experiment.

| ID | Requirement |
|---|---|
| PORT-001 | Clone and run on Windows/Linux with Python>=3.11; qualify observed platforms explicitly. No absolute developer paths, user-site packages, gorti server, C0 folder or external working tree required. |
| PORT-002 | Use a dedicated local venv and hash-locked dill0.4.1. Preserve reviewed native PyJevSim bytes/pins and licenses; package metadata is not engine qualification. Setup and network operations are outside measured time. Help is read-only. |
| PORT-003 | Default idle-primary is Q/M × N/C1 ×6 fresh paired families, L64/B8, suffix8/delta.25, balanced randomized method order. Fixed actions, not RL training. No C0 dependency. |
| PORT-004 | Compare actual independent physical projections; each method runs in a fresh sequential process. Preserve the existing application-wall kernel. Report CPU timing scope separately; no profiler or event counter in timing arms. |
| PORT-005 | Record OS/Python/source identity, user-declared condition and available host CPU/I/O observations. Missing coverage is null/unavailable; an idle label does not prove an interference-free host. Never pool host/OS cohorts automatically. |
| PORT-006 | New output directory only, bounded time/storage, no retry/replacement, failed/unexecuted denominators and process termination retained. Never terminate unrelated programs. Keep small receipts; no full successful snapshots/traces in Git. |
| PORT-007 | One simple entry point creates the environment, executes, analyzes and prints the result location. Default experiment cap600s/16MiB, owned attempt120s, worker1/BLAS1. No smoke/freeze/full-suite prerequisite or implicit experiment before user invokes it. |
| PORT-008 | Original gorti worktree/index/remotes and old evidence remain untouched. Import only scoped source files; record working-tree extraction and original commit rather than claiming a clean commit export. Preserve bytes through Git checkout. |

CLI may explicitly raise the default to at most7200s/128MiB; no implicit extension.
Linux Python installations must provide venv/ensurepip, or an already prepared,
dedicated venv. Missing distribution packages produce guidance, not a privileged
system installation. Python>=3.11 is the supported implementation target; actual
versions exercised must be listed separately in the implementation record.

## STD (direct verification, no execution gates)

| Case | Requirement | Direct check |
|---|---|---|
| PORT-T01 |001/002/007| Help creates no venv/output; path-with-spaces and interpreter selection; pinned dependency setup failure is explicit. |
| PORT-T02 |003/004| Pure manifest:24 unique arms/12 cells, six pairs/model, new seeds, three N-first and three C1-first pairs/model; paired config identical. |
| PORT-T03 |004/006| Direct small Q/M N/C1 kernel executions compare projections; failure and existing-output rejection retain denominators. These are functional checks, not timing evidence. |
| PORT-T04 |004/005| CPU/wall and optional host observations carry scope/availability; unavailable counters do not become zero; host condition is not a clean-host certificate. |
| PORT-T05 |006| Analysis known-ratio fixture, missing/failed/duplicate/mismatched rows, no-retry and exact-cell admission;24/12 denominators preserved. |
| PORT-T06 |001/002/008| Local clone checkout preserves pinned native/source bytes; no developer paths or prior-cohort files in executable configuration. Windows actual execution; Linux only if a local Linux runtime is available. |

Tests are development verification of changed transport/packaging and may be run
directly. `run_experiment.py` must not invoke tests, staging checks, source freeze
tools or an unrequested full experiment before the selected experiment.

## SDD

The imported `src/pyjevsim_bridge`, `src/rti1516e`, `vendor/pyjevsim` and the
original `bench/continuation_study/{cases,run,native_baseline}.py` remain byte-
preserved. Only new portable orchestration is implemented. The bridge's top-level
import needs SDK modules, but no RTI service/connection or generated network stubs
are used. The local benchmark needs dill, not the full gorti installation profile.

`run_experiment.py` selects `.venv/Scripts/python.exe` on Windows or
`.venv/bin/python` on POSIX, installs only the locked dependency when needed,
then invokes the benchmark with `-I -B`. A setup-only option is permitted.
`bench/design.py` constructs a manifest from `configs/idle-primary.json` and
explicit CLI overrides. Q961000/M962000+family0..5, planning961251,
bootstrap961252(+model offset); optional declared seed offset creates a distinct
cohort without changing the workload or selecting favorable results.

`bench/runner.py` owns one worker at a time; `bench/worker.py` uses repository-
relative source roots. N uses the native-correct sidecar; C1 maps to the existing
kernel's C backend. Full physical vectors travel through an owned pipe, are
compared in parent RAM, then discarded. Shared model/measurement provenance is
recorded. Fresh-process startup/import/transport are outside application wall but
inside campaign wall. CPU time sampled at backend-entry through kernel return
has its own explicit endpoint; do not subtract it from a differently scoped wall
metric and label the difference as pure waiting time.

`bench/host.py` reads optional start/end host CPU and I/O counters without polling
threads. Platform availability and aggregate scope are explicit. No claim that
host deltas isolate competing processes or identify the cause of slowdown.
`bench/analyze.py` reports modelwise wall and CPU ratios,20,000 paired-family
bootstrap95%CI, first3/last3 and method-order summaries, exact/failed/unexecuted
counts. This is exploratory, not a confirmation or a Windows-vs-Linux causal test.

## IDD

- CLI: `python run_experiment.py --preset idle-primary --condition idle`.
  Options: `--output`, `--budget-seconds`, `--max-mib`, `--seed-offset`, `--setup-only`.
- `bench.design.make_manifest(config: dict) -> dict`; config is the JSON preset
  plus validated CLI overrides. Manifest `arms/cases` retain the original kernel
  fields; methods `[N,C1]`; only purpose `timing`; `budget={seconds,max_bytes,attempt_seconds}`.
- `bench.runner.run_campaign(root, campaign, manifest) -> execution`; public
  `bench.runner.main(argv=None)` is the venv entry point. Implementation owners
  may agree on equivalent internal argument names without changing the CLI.
- Result files: `manifest.json`, `environment.json`, `arms.jsonl`, `cells.jsonl`,
  `execution.json`, `analysis.json`, `findings.md`, `analysis-status.json` and small
  receipts/provenance. Final success requires complete execution and analysis.

## Milestone/tasks, ownership and PDRR

MS-RL-36 / TASK-RL-200: portable repository and execution script. Root owns source
import, design/config, documentation, integration and commit. Runner agent owns
runner/worker; environment agent owns bootstrap/lock; analysis agent owns
analysis/host. Agents may develop in parallel; actual model checks run sequentially
after edits settle. No full24-arm performance experiment is requested in this task.

Plan is this document; Do/Review/Reflect and direct-check counts are recorded in
`docs/IMPLEMENTATION.md`. Same-purpose corrective task additions currently0;
stop at10 under the standing project rule. MS34's prior performance no-go and
MS35's unselected assurance policies are unchanged by this portability task.
