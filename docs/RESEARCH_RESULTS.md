# Continuation research results

The engineering research ran on Windows11 with CPython3.14.0, one worker and
BLAS1 from implementation commit82e7006. Semantics and new-model transfer
completed in the first workflow. Its runtime study was incomplete; after an
explicit user-approved time amendment, a separate second cost cohort completed
all330 arms. Neither cohort is pooled with the other or with historical Linux
results. Human productivity remains unmeasured.

핵심 판정: 복원 후 개입의 의미 보존과 새 모델의 공통 코어 재사용은 시험한 범위에서
확인했다. 앞부분 재실행 생략도 실제로 발생했다. 그러나 C1의 실행시간 우위와 개발
부담 감소는 입증하지 못했다. 후자는 4명 참여자 파일럿의 실제 관측이 필요하다.

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
below16MiB given the retained first-cohort size. The second cohort completed
in763.790 seconds including4.332 seconds of analysis, with no failed or unexecuted
arms. All324 timing arms and6 counting arms succeeded;108/108 timing and2/2
counting cells were exact. Each model has six complete nine-condition families.
All330 successful arms have one recorded source identity with zero source issues.
An independent read-only reviewer recomputed all54 geometric time ratios from
the saved rows; the maximum absolute difference from saved values was7.11e-15.

## Complete execution cost surface

Use `results/research-20261004-02/cost/analysis.json` for the complete cost study.
Ratios below1 favor the numerator. Every condition below has six paired families;
these are paired geometric ratios, not ratios of arithmetic mean times.

| Prefix and branches | Q N over R | Q C1 over R | M N over R | M C1 over R |
|---|---:|---:|---:|---:|
|16 and1|2.6265|36.7599|6.2078|59.3513|
|16 and4|1.3918|25.4754|2.5267|43.3606|
|16 and16|0.9660|19.7298|1.4369|34.5750|
|64 and1|2.0784|34.4578|3.2814|47.1904|
|64 and4|0.7734|14.5396|1.1042|20.4610|
|64 and16|0.3955|8.8778|0.4942|12.4678|
|256 and1|1.5463|40.0853|1.4996|31.7712|
|256 and4|0.4724|11.9418|0.4265|9.1915|
|256 and16|0.2023|4.8623|0.1638|3.3004|

At the largest exploratory grid condition L256/B16, not a retrospectively chosen
primary endpoint, the20000 whole-family bootstrap pointwise95 percent intervals
are:

| Model | N over R | C1 over R | C1 over N |
|---|---|---|---|
|Q|0.2023 [0.1912,0.2123]|4.8623 [4.4916,5.4795]|24.0301 [22.2994,26.5172]|
|M|0.1638 [0.1485,0.1837]|3.3004 [3.0110,3.7000]|20.1490 [19.3688,20.8844]|

N shows a discrete observed crossing between B1 and B4 at L256 for both models.
At L64, Q's N/R interval is below1 at B4/B16 and M's at B16. Q L16/B16 has a
point estimate slightly below1 but its interval includes1. No global/interpolated
threshold or simultaneous confidence statement follows from this grid.
C1/R and C1/N point estimates exceed1 in all18 model/condition combinations.
The C1/R pointwise interval lower bounds also exceed1; this is descriptive
exploratory evidence, not a preregistered confirmatory familywise test.
At L256/B16 the direction is unchanged in first/last-three and relative-method
order summaries. Q L16/B16 N/R changes from0.9127 in the first three to1.0224
in the last three, reinforcing its uncertain boundary. Each complete three-method
permutation has only one observation per condition; order effects are descriptive.

## Saved work and added runtime cost

The separate L256/B16 counting runs confirm R executes the prefix16 times
(4096 decision steps), while N and C1 execute it once (256 steps). N and C1
have identical model callback vectors. The actual whole-arm R minus N or C1
callback differences are:

| Model | Internal transitions | External transitions | Output callbacks | Confluent transitions |
|---|---:|---:|---:|---:|
|Q|3825|6705|3825|480|
|M|9570|14085|9570|525|

These are callback counts, not mutually exclusive unique simulated events.
M's whole-arm difference includes the avoided fresh-initialization callbacks;
prefix-only differences are9525 internal/output,14025 external and525 confluent
callbacks. The suffix vectors agree between all three methods. No model callbacks
were observed during capture/restore in these counting runs.

Mean application times and principal phase totals at L256/B16 are in seconds:

| Model and method | Application | Prefix | All restores | Suffix |
|---|---:|---:|---:|---:|
|Q R|1.1155|1.0129|not applicable|0.0184|
|Q N|0.2264|0.0651|0.0661|0.0209|
|Q C1|5.4697|2.5676|1.9433|0.7556|
|M R|1.7572|1.6526|not applicable|0.0455|
|M N|0.2881|0.1151|0.0782|0.0451|
|M C1|5.7955|2.9051|1.8456|0.8707|

The prefix saving mechanism is real, but one C1 prefix alone costs more than
all16 native-replay prefixes in this condition. C1 also incurs substantially
higher restoration and suffix costs. This localizes the cost without claiming
that this unprofiled run identifies the exact source-function cause. C1's mean
worker CPU is5.2969 seconds for Q and5.6068 for M, versus matched wall5.4698 and
5.7956; pure scheduler waiting cannot by itself explain these gaps. Cache,
frequency, I/O and other host effects are not causally isolated.

## Scope and retained evidence

The Q timing branches repeat the same normal-mode action sequence; M uses two
parity-selected action sequences repeated across branches. B16 therefore does
not mean16 distinct counterfactual interventions, and no full-suffix memoization
baseline was evaluated. Actual contrasting interventions were tested separately
in the semantics and inventory studies. A diverse-action performance extension
would be a separate predeclared study, not a relabeling of this data.
In M, B1 contains only the no-maintenance plan, while B4/B16 contain a50/50
mixture, so changing B also changes action composition. Increasing L changes
both prefix computation and accumulated history/state size. The present grid
does not independently identify prefix cost, state size and validation cost.

The complete cost cohort retains1,893,753 bytes; both cohorts together retain
3,423,705 bytes, below the shared16MiB ceiling. Source, worker cleanup and exact
comparison checks succeeded for the second cohort. Full lifetime memory and
host cleanliness remain unverified; clean-only sensitivity is unavailable.
Source inventory is not a complete dependency/environment certificate.
The20,000-resample intervals are model-specific whole-family, pointwise and
exploratory. Six families are not six independently sampled application domains.
No training convergence, learned-policy quality, federation or general parallel
speedup was evaluated. No source or results were pushed or published by this task.

## Human evaluation status

Four potential senior undergraduate participants have agentic AI development
experience and no simulation experience. DEVELOPER_STUDY.md defines a novice
AI-assisted pilot with balanced treatment/task assignment and equal AI access.
No participant sessions have been conducted. Matched task packs, consent,
scheduling and actual observations still need completion before an effort claim.
