# Conditional preservation of restored intervention semantics

The implementation supports continuation at declared committed fixed-delta
boundaries. This argument explains a sufficient condition for observational
equivalence; the finite experiments check selected instances of that condition.
It is not a machine-checked theorem or a guarantee for arbitrary model code.

## State and observation

Let X contain model state, shared object relationships, engine and wrapper
clocks, pending scheduler entries and their ordering, random generator state,
input/output boundary state, cached observation, reward accumulator and episode
termination state. Let u be a sequence of future external inputs together with
their logical injection times. Let O include ordered observable events, returned
observations, reward deltas, logical times and termination flags. Physical
instance identifiers may be renamed only by an explicit identity mapping.

## Sufficient conditions

1. The cut is supported and quiescent according to its declared boundary. No
   partially executed callback or undeclared input remains to be reconstructed.
2. Snapshot captures every value that can influence future transitions or O,
   including RNG, aliases, scheduler ordering and RL boundary caches.
3. Restoration reconstructs that state up to the allowed identity mapping. It
   consumes no simulation event or random draw and does not reset the model.
4. Transition functions and scheduler semantics remain the same. Time precision,
   simultaneous-event ordering and action delivery ordering are preserved.
5. Each path receives the same logical future input sequence u, including the
   first intervention at the restored cut. Future randomness is coupled through
   the restored PRNG state, not merely described as having the same seed.
6. Uncaptured external state, wall-clock dependence and shared mutable state
   outside the declared ownership model do not influence these transitions.

## Inductive argument

Immediately after restoration, condition3 establishes equality of all relevant
state under the allowed identity mapping. If the pre-transition states agree,
conditions4 and5 select the same next event/input and the same random value.
The same transition therefore produces corresponding successor states and
outputs. Induction establishes equal finite event/output sequences. Because
the reward baseline and termination state belong to X, RL step observations,
reward deltas and termination also agree. Condition6 excludes effects not
covered by that induction. It is precisely an assumption to assess, not a
property inferred from the word snapshot.

## Intervention boundary

The first post-restore action enters at the restored logical time before the
next environment advance. This does not insert an action before events already
committed at that time before capture. General arbitrary-port event bags,
retroactive injection, unbounded zero-time loops, unsupported dynamic topology
and arbitrary external I/O are not covered by this argument.

## Connection to measured evidence

The Q/M study compares N and C1 to fresh replay, checks direct clocks/state,
ordered histories, RNG continuation in M, reward caches, alias relations and
noninterference of simultaneously live branches. It observes zero model
callbacks during capture/restore and action insertion/delivery at the cut.
The inventory study adds an independent domain oracle and a maintained state
extension. Negative controls establish sensitivity of selected observers or
validators; they do not prove complete detection of all implementation defects.

All three methods still use the same native simulator for Q/M, so correlated
simulator/model defects remain possible. Inventory's separate event oracle
reduces that particular dependence for its declared deterministic scenarios.
Source verification is not proof that model state declarations are complete.
The distinction between sufficient conditions, implementation obligations,
finite observations and untested generalization must remain in the manuscript.
