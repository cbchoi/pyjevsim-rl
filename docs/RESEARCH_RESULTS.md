# Continuation research results

The first engineering research workflow ran on Windows11 with CPython3.14.0,
one worker and BLAS1 from implementation commit82e7006. Semantics and new-model
transfer completed. Runtime measurement stopped at its declared deadline and
is not a complete cost study. Historical Linux results are not pooled.

## Completed semantic evidence

`results/research-20261004-01/semantics.json` records18/18 successful conditions,
162 branch trajectories, all restoration-versus-replay comparisons exact and
all five disposable-native corruption controls detected. Observed model callback
counts during capture/restore were zero. First post-restore action insertion
and delivery occurred at the restored logical cut. Simultaneously live branches
and the captured source remained isolated. Q's three seeds reuse one explicit
arrival tape; M exercises future repair RNG. This is finite conformance evidence,
not a proof for arbitrary models or independent simulator implementations.

## Completed new model evidence

`results/research-20261004-01/transfer.json` records54/54 exact three-method
branch cells,162/162 primary executions,36/36 repeated-branch diagnostics and
six correctly scoped negative controls. Results match both native replay and
a separately implemented inventory event oracle. Cut0.75 contains an actual
outstanding order of5 due at1.0. V2 penalty and incremental reward state are
preserved. The imported continuation core, native sources and original kernel
are unchanged relative to b3c7251.

New code inventory is243 physical lines of shared model/binding code,31 lines
for V2,269 lines for the C1 adapter and179 lines for N's sidecar. Counts include
comments/blanks and unequal assurance responsibilities; N also reuses existing
native-baseline helper code. Thus this case does not establish less coding for
C1. It establishes reuse of an unchanged core and explicit state/reward ownership.
Human development time, productivity and maintenance-time savings are unmeasured.

## First cost cohort stopped by budget

`results/research-20261004-01/cost/execution.json` records230/324 attempted timing
arms:229 succeeded, one was terminated on the campaign deadline,94 unexecuted.
The six separately instrumented counting arms were not reached. All76 completed
comparison cells were exact out of108 planned. Four complete nine-condition
families per model were observed out of six planned. No retry/replacement occurred.
The full workflow elapsed579.612 seconds including9.528 seconds of completed
partial analysis. Recorded sources agreed across229 successful arms.

At L256/B16, the incomplete cohort's paired geometric N/R ratios were0.1906 for
Q and0.1599 for M; C1/R ratios were4.4477 and3.7038. These are partial exploratory
observations, not a full-grid boundary conclusion. No observed C1/R point estimate
was below1, whereas native reuse showed gains in several conditions. Actual
callback savings are unmeasured because counting arms were not executed. Decision
step counts must not be relabeled as unique simulated events.

The first cohort retains1,529,952 bytes. Boundary-observed semantic and transfer
temporary peaks were50,822 and3,708 bytes. Unobserved transient peaks, whole-lifetime
memory and interference-free host status remain unknown. Successful complete
traces are ephemeral; retained hashes do not reconstruct them.

## Approved second cost cohort

After the first cohort stopped, the user approved a separate1200-second cost-only
cohort. Keep the same324 timing plus6 counting arms, inputs, seeds, methods,
analysis and source implementation. Do not pool either cohort or replace the
first failed arm. The reason is the insufficient execution budget, not favorable
results. Set the new result cap to14MiB, keeping both research cohorts together
below16MiB given the retained first-cohort size. Final results are pending.

## Human evaluation status

Four potential senior undergraduate participants have agentic AI development
experience and no simulation experience. DEVELOPER_STUDY.md defines a novice
AI-assisted pilot with balanced treatment/task assignment and equal AI access.
No participant sessions have been conducted. Matched task packs, consent,
scheduling and actual observations still need completion before an effort claim.
