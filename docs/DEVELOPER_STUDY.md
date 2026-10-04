# Developer integration and maintenance evaluation protocol

The user reports that four senior undergraduate students may be available.
They have agentic AI development experience and no simulation experience.
This is a novice, AI-assisted procedural pilot, not a powered general-developer
productivity experiment. No task sessions have been collected, and no
developer-time reduction has been established. The automated inventory case
study is structural evidence and must not be represented as a human experiment.

## Questions and treatments

Does C1 reduce time to a correct integration compared with handwritten native
snapshot restoration N? Does it reduce time to a correct model-state change
while preserving previous behavior? Both treatments receive working native
PyJevSim, equivalent documentation, starter model code and parameterized action
input. Neither treatment edits source per branch. C1 adapters count as work.

Define common required behavior separately from optional framework guarantees.
If source identity or malformed-state rejection is required, specify that
upfront for both treatments. Label an extended N implementation distinctly from
the existing fast N; do not compare unequal requirements as though equivalent.

## Participants and assignment

Recruit developers independently of the framework authors. Record Python,
discrete-event simulation and snapshot implementation experience. Obtain informed
agreement and any institution-required ethics review before collecting sessions.
Do not collect unnecessary personal identifiers.

Use Q/M for training. Select two new matched task models not exposed in training
or existing solutions. The completed inventory model cannot subsequently be
called an untouched held-out task. A participant receives different matched
model/task variants in the two treatments. Counterbalance treatment and variant
order; record assignment before sessions. Equalize training, reference materials,
hardware, permitted assistance and AI access. Analyze participant and task
effects; repeated observations from one developer are not independent people.

## Task objects and outcome definitions

| Task | Fixed deliverable | Correctness evidence |
|---|---|---|
| Initial integration | Capture committed state, restore independent branches, apply a parameterized new action | Independent replay comparison of clock, ordered outputs, reward, state and branch isolation |
| State extension | Add a persistent accumulator and incremental reward baseline | Existing regression checks plus new-state continuation correctness |
| Shared resource change | Modify an aliased resource or pending event relationship | Preservation of alias, schedule and post-restore effects |

Primary endpoints are elapsed time to the first correct submission and the
proportion completing within a fixed task budget. Record active work time,
unsuccessful submissions, defects, assistance and rework separately. Do not
discard failed participants or replace a timeout with a successful completion
time. Treat incomplete times as censored and report completion rates explicitly.
Use correct paired/repeated-measures analysis for the final assignment design.
Code size and number of touched files are secondary structural outcomes only.

## Four participant pilot

Use anonymous assignment slots, assigning the four consenting people to slots
randomly before work. A and B are matched task variants whose final instructions
must be completed and reviewed before sessions; they are not the existing
inventory solutions distributed as a fresh implementation task.

| Slot | First period | Second period |
|---|---|---|
| P01 | C1 with A | N with B |
| P02 | N with A | C1 with B |
| P03 | C1 with B | N with A |
| P04 | N with B | C1 with A |

Provide the same 30 minute introduction to simulation clock, event ordering,
snapshot cuts, state ownership and reward deltas, then a 15 minute practice
using a separate model. Plan two sessions: the first includes this orientation,
a break, and a 60 minute task period; the second includes another 60 minute
period and a short interview. Within each period allocate 40 minutes to initial
integration and 20 minutes to maintenance. These are pilot budgets to test
feasibility, not promises that novices can finish. If integration is incomplete,
retain that failure and provide a standard correct baseline for maintenance,
recording the assistance; do not remove that participant from the second task.

Keep AI access identical across treatments: model/version, tools, network policy,
initial instructions, token/spending limits, reference access and permitted
researcher help. Use a fresh AI context and workspace for each period; record
the actual prompts, tool actions and generated patch under the consent policy.
Do not permit one treatment to reuse the other treatment's answer. Equal AI
settings reduce one confound; they do not make participants or generated
solutions independent. Report human planning/review/debugging and AI-assisted
elapsed time as separate measures where observable.

With four participants, show all individual outcomes, task failures, order and
experience. Paired differences are descriptive, not a population-level proof.
Do not treat eight periods, commits, AI agents or repeated tests as eight or
more independent participants. A four-person pilot may find protocol problems
and feasibility limits even if it cannot support a reliable superiority claim.

## Pilot and analysis commitments

The procedural pilot checks task difficulty, timing capture, rubric ambiguity,
learning effects and timeouts. Set confirmatory sample size from a justified
minimum useful effect and observed variance/precision needs; there is no
arbitrary universal minimum participant count. Pilot observations do not
automatically become confirmation data. Record hypotheses, task budgets,
exclusions, endpoints, assignment, sample calculation and analysis before the
main study. Report effect sizes and intervals, all planned outcomes and failures.
Where practical, have an independent assessor evaluate correctness without
seeing treatment labels. Document when blinding is impossible from source form.

## Session receipt

Retain pseudonymous participant ID, experience categories, assignment, task and
model version, source hashes, start/end/idle intervals, permitted tool use,
submission times, correctness decisions, assistance, timeout/withdrawal status,
and consent/retention policy reference. These fields describe future collection,
not completed observations. Keep personal information and raw recordings out of
Git. No recruitment, public upload or private-data collection is authorized by
the automated research implementation.

## Interpretation

The current engineering study may establish that the generic core stayed
unchanged and that a particular new model and state extension behaved correctly.
It cannot establish saved human minutes, population-level usability, or universal
model support. Runtime cost is evaluated separately. A total economic cost model
must state developer and compute prices, model/change frequency and execution
count; raw developer-hours and CPU-hours are not interchangeable units.

Methodological reference: [ACM SIGSOFT Empirical Standards](https://www2.sigsoft.org/EmpiricalStandards/docs/standards).
The task objects and measurement protocol above are this project's proposal,
not prescribed tasks or acceptance guarantees from that reference.
