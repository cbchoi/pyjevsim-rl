"""Explicit ownership composition, not discovery of hidden Python state.

semantic_path is a reviewed canonical ownership key. Aliased spellings are a
model-review obligation; this helper cannot prove closure of arbitrary code.
"""
from __future__ import annotations

from .contracts import StateObligation, fail


def compose_obligations(*, engine=(), model=(), boundary=(), coordinator=()):
    """Combine disjoint ownership declarations without upgrading evidence."""
    result = []
    identifiers = set()
    paths = set()
    for owner, entries in (("engine", engine), ("model", model),
                           ("boundary", boundary), ("coordinator", coordinator)):
        if type(entries) not in (tuple, list):
            fail("ownership groups must be explicit tuple/list declarations")
        for entry in entries:
            if type(entry) is not StateObligation or entry.owner != owner:
                fail(f"{owner} declaration has a different or invalid state owner")
            if entry.obligation_id in identifiers:
                fail("duplicate obligation ID across ownership groups")
            if entry.semantic_path in paths:
                fail("canonical state path has multiple ownership declarations")
            identifiers.add(entry.obligation_id)
            paths.add(entry.semantic_path)
            result.append(entry)
    return tuple(result)


def _entry(number, path, owner, read, write, rule, invariant, counterexample, tests):
    return StateObligation(
        obligation_id=f"OB-CC-{number}", semantic_path=path, owner=owner,
        read_sites=tuple(read), write_sites=tuple(write), restore_rule=rule,
        invariant=invariant, counterexample=counterexample,
        test_ids=tuple(tests), evidence_status="reviewed",
    )


def engine_obligations():
    """Common obligations for the installed restricted flat HLA provider."""
    return (
        _entry("E01", "engine.clock", "engine", ("SysExecutor.step",),
               ("SysExecutor.step",), "restore executor time without advancing",
               "engine time equals committed boundary time", "clock-only omission",
               ("TC-CC-004", "TC-CC-007")),
        _entry("E02", "engine.behavior_schedule", "engine",
               ("BehaviorExecutor.set_req_time", "ScheduleQueue"),
               ("BehaviorExecutor.set_req_time", "SysExecutor._run_instant"),
               "preserve behavior/wrapper clocks and live native calendar",
               "no inferred residual; model tie obligation remains explicit",
               "nonaligned cut and simultaneous events", ("TC-CC-004", "TC-CC-005")),
        _entry("E03", "engine.topology", "engine", ("SysExecutor.route",),
               ("EngineStateProvider.attach",), "reconstruct declared routing only",
               "static-flat declared nodes and native confluence; no hidden queues",
               "changed coupling or unsupported hierarchy", ("TC-CC-006", "TC-CC-014")),
    )


def boundary_obligations():
    """Model-independent cursor/cache obligations; model validates projections."""
    return (
        _entry("B01", "boundary.commit_cache", "boundary",
               ("PyJevSimEnv._step_unlocked",), ("PyJevSimEnv._step_unlocked",),
               "restore complete environment payload without reset or callbacks",
               "nonterminal committed cut; exact cursor/cache/seed/horizon",
               "premature driver-only cut or stale reward observation",
               ("TC-CC-003", "TC-CC-007", "TC-CC-012")),
        _entry("B02", "boundary.reward_handle", "boundary",
               ("model reward callback",), ("model reward callback",),
               "restore candidate BoundaryStateHandle values in place",
               "binding and provider share candidate handle only",
               "reward baseline reset or sibling alias",
               ("TC-CC-009", "TC-CC-012")),
        _entry("B03", "boundary.lineage", "boundary",
               ("fixed-policy caller",), ("explicit BranchContext",),
               "preserve family/prefix/policy and declared suffix lineage",
               "physical worker slot cannot change logical identity",
               "policy or sampling generation substitution",
               ("TC-CC-010", "TC-CC-013")),
    )


def coordinator_obligations():
    return (
        _entry("C01", "coordinator.reference_ownership", "coordinator",
               ("ReferenceRegistry",), ("adapter.rebind", "engine.attach"),
               "rebind candidate-local references with stable semantic IDs",
               "no source/sibling mutable object or bound-clock alias",
               "shared RNG/resource across branches", ("TC-CC-009", "TC-CC-014")),
        _entry("C02", "coordinator.failure_atomicity", "coordinator",
               ("capture", "restore", "RuntimeHandle.close"),
               ("restore candidate", "CleanupLedger.close"),
               "publish only fully validated candidate; close only owned resources",
               "failed capture preserves source or marks invalid; cleanup errors persist",
               "failure during allocate/rebind/validate and failing release",
               ("TC-CC-007", "TC-CC-008", "TC-CC-015")),
    )
