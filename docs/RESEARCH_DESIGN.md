# Continuation semantics reuse and execution cost study

This study evaluates three distinct claims: preservation of restored execution
and intervention semantics; integration and maintenance of a new model without
changing the continuation core; and the runtime cost boundary between replay R,
handwritten native snapshot N, and the common continuation framework C1.
It is an exploratory, AI-authored engineering study, not a human productivity
experiment, a confirmation study, or evidence of universal novelty or speedup.
The existing Linux results and imported implementation remain unchanged.

## Requirements and test design

| Requirement | Required evidence | Direct test or experiment |
|---|---|---|
| RES001 | Independent fresh replay agrees with restored N and C1 after the same interventions | Q/M each three seeds and cuts 1/4/9, two contrasting action branches, stepwise physical comparisons |
| RES002 | Restore is a state operation, not simulated catchup; the next action enters at the restored clock | Capture/restore callback counts, injection timestamp, logical clock, RNG/reward state and shared references |
| RES003 | Branch effects are physical and branches do not contaminate each other | Compare observations/history without branch labels; repeat branch A after B; check captured source unchanged |
| RES004 | The equality observer detects plausible defects | Deliberate clock/reward/RNG/event-order differences; label observer controls separately from live-runtime corruption |
| RES005 | A new domain integrates without modification of the generic core or native sources | Inventory/replenishment V1 model, declared C1 adapter, handwritten N restore, independent replay/event oracle |
| RES006 | A concrete model change has explicit state ownership and survives restoration | V2 stockout-penalty accumulator and reward baseline, regression and omission controls, version compatibility |
| RES007 | Integration effort evidence does not masquerade as developer productivity | Source/change inventory and responsibilities; human time, independent participants, causal effort reduction remain unmeasured |
| RES008 | R/N/C1 execute the same workload and physical outcomes | Fresh sequential process per timing arm; exact three-way physical equality before accepting a cell |
| RES009 | Measure an execution cost surface rather than select a favorable cell | Q/M x prefix16/64/256 x branches1/4/16 x six paired families x R/N/C1; suffix4, delta0.25 |
| RES010 | Retain unfavorable, failed, unexecuted and uncertain evidence | No retries/substitution, bounded run, compact receipts, bootstrap/order sensitivity, memory and host uncertainty |
| RES011 | Reproduction remains portable and simple | One research entry point, existing dedicated venv, no smoke/freeze/staging gate or new runtime dependency |

The scientific equality checks above are measurements of the requested claim,
not ancillary execution gates. Only directly relevant development tests run.
No external program is stopped, no historical result is overwritten, and no
repository is pushed or published by this task.

## Semantics study

Q uses normal versus fast service; M uses no maintenance versus maintenance at
the first suffix action, including repair jitter. Seeds 971000/971001/971002
and cuts 1/4/9 give 18 model/seed/cut cases. Each compares R/N/C1 under the same
input history and future action sequence, with 16 suffix steps. Stable logical
identity is checked; physical instance names are normalized only where needed.
Equal end rewards alone are insufficient: retain compact digests and explicit
pass/fail receipts for step observations, ordered history, reward, clocks, RNG,
shared aliases, source immutability and A/B/A branch isolation. Instrumentation
is confined to this non-timing study.

The claimed cut is a committed fixed-delta boundary with its required inputs
drained. A newly injected action is applied to that restored boundary, not
retroactively before events already committed at the same logical timestamp.
Under complete state capture, preserved future inputs/RNG, unchanged transition
functions and preserved scheduler ordering, restoration should be observationally
equivalent to uninterrupted continuation. Finite tests support this property
for the tested profiles; they are not a proof for arbitrary Python models.

## New model and maintenance study

Use a new inventory/replenishment model with a demand source and stock actor,
pending replenishment events, conservation identities and declared action input.
Implement V1 and an explicit V2 addition of cumulative stockout penalty and its
incremental reward baseline. C1 uses the existing generic bundle and boundary;
N uses native snapshot data and handwritten clock/calendar/boundary sidecar;
R reconstructs the prefix. A separate domain oracle reduces correlated errors.
The fixed cases are V1/V2 x deterministic configurations 0/1/2 x cuts0/3/4
at delta0.25 x first-suffix order quantities0/4/9, followed by seven zero
actions. These are 54 three-method branch cells and 162 branch executions,
not 54 independent stochastic replications. Prefix orders quantity5 at step
index1; demand times are0.25/0.75/1.5/2.25/3.0 and replenishment lead0.5.
Include time0 cuts, in-flight orders, demand/replenishment ties, repeated-branch
isolation, and malformed V2 penalty/baseline rejection. Record these settings
in the result manifest. No Q/M/packet code is relabeled as an unseen model.

Inventory is new relative to the preexisting framework, but its implementation
is developed by the same AI team with access to both methods. Therefore it is
not a blinded or prospectively held-out human study. Measure source ownership,
changed functions/files and correctness of the V1-to-V2 change, not fabricated
human minutes. Count shared model code once; adapter code is not zero-cost.
An AST/source inventory is structural evidence, not an effort estimator.

A later human study needs independent participants, matched integration and
maintenance tasks, balanced assignment, equal tools/training, time-to-correct
completion and failed/time-limited denominators. Its sample size requires a
pilot and a precision/power justification. This task does not recruit people
or claim to complete that study.

## Runtime cost study

The predeclared grid has 324 timing arms and 108 three-method comparison cells.
Each model/family is one repeated-measures family across all nine conditions.
Seeds are Q971000..971005 and M972000..972005; planning seed973251;
bootstrap seed973252. All six permutations of R/N/C1 occur once per condition.
Randomize condition order within family. Use arrivals that continue through the
largest prefix plus suffix horizon; do not benchmark an accidentally empty model.
Action plans are parameterized for all methods, with no branch-specific source
editing charged only to N. C1 maps to the unchanged kernel's C method.

Report C1/R, N/R and C1/N paired geometric wall ratios, absolute paired times,
phase components and separately scoped CPU/process wall metrics. Use 20000
whole-family resamples with pointwise exploratory 95 percent intervals; do not
interpret the collection of cells as simultaneous confidence or confirmation.
Report relative-order and first/last-three-family sensitivity descriptively.
An observed crossing is a grid observation, not a guaranteed global threshold.

Measure native callback counts separately after timing at L256/B16, family0,
for Q/M x R/N/C1 (six counting arms). Callbacks and decision steps are not
unique simulation events: report the callback vector by type and actual prefix
invocations, never relabel their sum as an independent event count.

Use the cost identity T_R = B(P_R + S_R) and
T_C1 = P_C1 + capture + B(restore_C1 + S_C1), with setup/cleanup and other
measured work added when predicting end-to-end time. If the per-branch saving
is nonpositive, increasing branches alone does not guarantee a crossover.
Any fitted/pilot cost prediction is exploratory; it is not independently validated
unless a separate predeclared validation sample is actually collected.

## Interfaces and execution limits

`bench.research.semantics.run_study` returns a compact semantics report.
`bench.research.transfer.run_transfer` returns the new-model engineering report.
`bench.research.cost` constructs the runtime design and configures the existing
owned-process worker/kernel; `cost_analysis` summarizes the three-way results.
`run_research.py` provides semantics, transfer, cost and all stages, and writes
fresh result directories only. Research modules are recorded as executed source.

Default research workflow limits are 600 seconds including analysis, 16 MiB new
results/transient storage, one simulation worker and BLAS1, 120 seconds per
timing arm. Development and directly relevant tests are outside timing evidence.
Reserve the final30 seconds of the remaining workflow time for cost analysis;
the simulation submission budget excludes this fixed reserve. Budget expiry
preserves partial denominators and stops; no automatic extension
or rerun. Memory lifetime and interference-free host status remain unconfirmed.
Successful full traces/snapshots are ephemeral; compact findings, per-case
receipts, source identities and failure records remain. Results from this host
are not pooled with the user's earlier Linux cohort.

## Milestones tasks and traceability

| Milestone | Task | Requirement coverage | Owned implementation |
|---|---|---|---|
| MS37 | TASK201 Plan | RES001..011 | This design, protocol and interfaces before implementation |
| MS37 | TASK202 Semantics | RES001..004 | semantics.py and direct tests |
| MS37 | TASK203 Transfer | RES005..007 | inventory model/adapter/native implementation, transfer.py, direct tests |
| MS37 | TASK204 Cost | RES008..010 | cost.py, cost_analysis.py, direct tests |
| MS37 | TASK205 Integration and results | RES010..011 and all evidence | CLI, sequential research execution, independent review, results |

Each task follows Plan, Do, Review, Reflect in docs/RESEARCH_IMPLEMENTATION.md,
with loop numbering MS37-L1 etc. Corrections for the same purpose stop at ten
added tasks. Commit completed task paths explicitly as cbchoi with claude
<me@cbchoi.info>; preserve unrelated changes and imported source bytes.
