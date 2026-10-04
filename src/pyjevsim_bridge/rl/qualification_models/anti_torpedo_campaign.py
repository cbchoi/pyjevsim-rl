"""Local learnability campaign for the prototype anti-torpedo workload.

The campaign is deliberately claim-bounded.  It executes real local AT/SIM
episodes, learns a scenario-conditioned action table from those episodes, and
compares the learned policy with frozen and deterministic-random floors on the
same holdout seeds.  The external AT/SIM source profile is not claim-grade, so
even a successful learnability witness always reports ``claim_grade=False``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Final

from pyjevsim_bridge.rl.executor import PINNED_PYJEVSIM_2_1_2_PROFILE
from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
    CLAIM_GRADE as ANTI_TORPEDO_CLAIM_GRADE,
)
from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
    ENVIRONMENT_CONTRACT_SHA256,
    LOADED_ADAPTER_SOURCE_SHA256,
    MAX_STEPS,
    PYJEVSIM_EXECUTOR_SOURCE_SHA256,
    SCENARIO_BANK_SHA256,
    anti_torpedo_environment_factory,
    atsim_source_sha256,
    scenario_config_sha256,
    scenario_for_seed,
)
from pyjevsim_bridge.rl.scientific import (
    EXCLUSION_POLICY_ID,
    RERUN_POLICY_ID,
    SEMANTIC_PROJECTION_CONTRACT_SHA256,
    ScientificExperimentManifest,
    ScientificProtocolError,
    canonical_semantic_projection,
    derive_global_episode_seed,
    semantic_projection_sha256,
)

CAMPAIGN_SCHEMA_VERSION: Final = "anti-torpedo-local-learnability-v1"
TRAINER_ID: Final = "deterministic-scenario-contextual-policy-selection-v1"
RANDOM_FLOOR_ID: Final = "sha256-step-random-floor-v1"
RESOURCE_PROFILE_ID: Final = "single-host-single-process-local-v1"
UNCERTAINTY_METHOD: Final = (
    "deterministic-sha256-paired-bootstrap-10000-two-sided-95-v1"
)
BOOTSTRAP_SAMPLES: Final = 10_000
CONFIDENCE_LEVEL: Final = 0.95
ACTIONS: Final = (0, 1, 2, 3, 4, 5)
SCENARIOS: Final = ("self_propelled", "stationary")
EVALUATION_POLICIES: Final = ("trained", "frozen", "random")
PRIMARY_ENDPOINTS: Final = (
    "mean_holdout_return",
    "trained_minus_frozen_paired_return",
    "trained_minus_random_paired_return",
)
DEFAULT_TUNING_SEEDS: Final = (10_000, 10_001)
DEFAULT_MEASURED_MASTER_SEEDS: Final = tuple(range(10))
DEFAULT_EVALUATION_SEEDS: Final = (2_000, 2_007)
DEFAULT_QUALITY_THRESHOLD: Final = 40.0
CLAIM_LIMIT: Final = (
    "Local prototype-source learnability evidence over two deterministic "
    "scenario literals only; repeated master seeds do not establish independent "
    "scenario generalization, tuning seeds are declared but not executed, and a "
    "degenerate paired interval is not informative uncertainty evidence. External "
    "replay/process prerequisite evidence is not bound into this result; not "
    "claim-grade, not a gorti/RLlib comparison, and not evidence of general "
    "learning superiority."
)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificProtocolError(
            f"campaign value is not canonical JSON compatible: {exc}"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _algorithm_profile() -> dict[str, object]:
    return {
        "actions": list(ACTIONS),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence_level": CONFIDENCE_LEVEL,
        "evaluation_policies": list(EVALUATION_POLICIES),
        "random_floor_id": RANDOM_FLOOR_ID,
        "scenario_generalization": False,
        "scenarios": list(SCENARIOS),
        "trainer_id": TRAINER_ID,
        "training_update": "one-exhaustive-return-per-scenario-action",
        "uncertainty_method": UNCERTAINTY_METHOD,
    }


def _resource_profile() -> dict[str, object]:
    return {
        "backend": "local-in-process",
        "environment_instances": 1,
        "profile_id": RESOURCE_PROFILE_ID,
        "worker_identity_in_seed": False,
    }


def campaign_implementation_sha256() -> str:
    """Return the content identity of this campaign implementation."""

    return _file_sha256(Path(__file__).resolve())


LOADED_CAMPAIGN_IMPLEMENTATION_SHA256: Final = campaign_implementation_sha256()
ALGORITHM_CONFIG_SHA256: Final = _digest(_algorithm_profile())
RESOURCE_BUDGET_SHA256: Final = _digest(_resource_profile())


def _verify_campaign_source_lock() -> None:
    if campaign_implementation_sha256() != LOADED_CAMPAIGN_IMPLEMENTATION_SHA256:
        raise ScientificProtocolError(
            "campaign implementation source changed after module admission"
        )


def build_local_campaign_manifest(
    *,
    tuning_seeds: Sequence[int] = DEFAULT_TUNING_SEEDS,
    measured_master_seeds: Sequence[int] = DEFAULT_MEASURED_MASTER_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> ScientificExperimentManifest:
    """Build the source-locked default manifest before any episode runs."""

    _verify_campaign_source_lock()
    tuning = tuple(tuning_seeds)
    measured = tuple(measured_master_seeds)
    evaluation = tuple(evaluation_seeds)
    training_step_budget = len(measured) * len(ACTIONS) * len(SCENARIOS) * MAX_STEPS
    evaluation_episode_count = (
        len(measured) * len(evaluation) * len(EVALUATION_POLICIES)
    )
    evaluation_step_budget = evaluation_episode_count * MAX_STEPS
    return ScientificExperimentManifest(
        experiment_id="anti-torpedo-local-learnability-v1",
        model_sha256=atsim_source_sha256(),
        plugin_sha256=LOADED_ADAPTER_SOURCE_SHA256,
        environment_contract_sha256=ENVIRONMENT_CONTRACT_SHA256,
        projection_contract_sha256=SEMANTIC_PROJECTION_CONTRACT_SHA256,
        learner_implementation_sha256=LOADED_CAMPAIGN_IMPLEMENTATION_SHA256,
        algorithm_config_sha256=ALGORITHM_CONFIG_SHA256,
        resource_budget_sha256=RESOURCE_BUDGET_SHA256,
        tuning_seeds=tuning,
        measured_training_master_seeds=measured,
        evaluation_seeds=evaluation,
        environment_step_budget=training_step_budget + evaluation_step_budget,
        evaluation_interval_steps=training_step_budget,
        evaluation_episodes=evaluation_episode_count,
        primary_endpoints=PRIMARY_ENDPOINTS,
        quality_threshold=quality_threshold,
        noninferiority_margin=0.0,
        exclusion_policy_id=EXCLUSION_POLICY_ID,
        rerun_policy_id=RERUN_POLICY_ID,
    )


@dataclass(frozen=True, slots=True)
class EpisodeRunPlan:
    """One worker-count-independent episode identity in the frozen plan."""

    run_id: str
    phase: str
    master_seed: int
    episode_seed: int
    global_episode_ordinal: int | None
    scenario_id: str
    policy_id: str
    fixed_action: int | None

    def content(self) -> dict[str, object]:
        return {
            "episode_seed": self.episode_seed,
            "fixed_action": self.fixed_action,
            "global_episode_ordinal": self.global_episode_ordinal,
            "master_seed": self.master_seed,
            "phase": self.phase,
            "policy_id": self.policy_id,
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
        }


@dataclass(frozen=True, slots=True)
class EpisodeRunReceipt:
    """Terminal receipt retained for every planned episode."""

    plan: EpisodeRunPlan
    status: str
    total_return: float | None
    steps: int
    outcome: str | None
    action_trace_sha256: str | None
    semantic_sha256: str | None
    atsim_model_source_sha256: str | None
    adapter_source_sha256: str | None
    environment_contract_sha256: str | None
    projection_contract_sha256: str | None
    scenario_bank_sha256: str | None
    config_sha256: str | None
    pyjevsim_executor_profile_id: str | None
    pyjevsim_executor_revision: str | None
    pyjevsim_executor_source_sha256: str | None
    executor_qualification_policy_id: str | None
    source_claim_grade: bool
    executor_qualified: bool
    error: str | None

    def content(self) -> dict[str, object]:
        return {
            "action_trace_sha256": self.action_trace_sha256,
            "adapter_source_sha256": self.adapter_source_sha256,
            "atsim_model_source_sha256": self.atsim_model_source_sha256,
            "config_sha256": self.config_sha256,
            "environment_contract_sha256": self.environment_contract_sha256,
            "error": self.error,
            "executor_qualified": self.executor_qualified,
            "executor_qualification_policy_id": (
                self.executor_qualification_policy_id
            ),
            "outcome": self.outcome,
            "plan": self.plan.content(),
            "projection_contract_sha256": self.projection_contract_sha256,
            "pyjevsim_executor_profile_id": self.pyjevsim_executor_profile_id,
            "pyjevsim_executor_revision": self.pyjevsim_executor_revision,
            "pyjevsim_executor_source_sha256": (
                self.pyjevsim_executor_source_sha256
            ),
            "scenario_bank_sha256": self.scenario_bank_sha256,
            "semantic_sha256": self.semantic_sha256,
            "source_claim_grade": self.source_claim_grade,
            "status": self.status,
            "steps": self.steps,
            "total_return": self.total_return,
        }


@dataclass(frozen=True, slots=True)
class _EpisodeProvenance:
    atsim_model_source_sha256: str
    adapter_source_sha256: str
    environment_contract_sha256: str
    projection_contract_sha256: str
    scenario_bank_sha256: str
    config_sha256: str
    pyjevsim_executor_profile_id: str
    pyjevsim_executor_revision: str
    pyjevsim_executor_source_sha256: str
    executor_qualification_policy_id: str
    source_claim_grade: bool
    executor_qualified: bool


@dataclass(frozen=True, slots=True)
class LearnedContextPolicy:
    """Scenario action table selected only from retained training returns."""

    master_seed: int
    selected_actions: tuple[tuple[str, int], ...]
    action_returns: tuple[tuple[str, int, float], ...]
    training_run_ids: tuple[str, ...]
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_sha256", _digest(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "master_seed": self.master_seed,
            "action_returns": [list(row) for row in self.action_returns],
            "selected_actions": [list(row) for row in self.selected_actions],
            "trainer_id": TRAINER_ID,
            "training_run_ids": list(self.training_run_ids),
        }

    def action_for(self, scenario_id: str) -> int:
        for scenario, action in self.selected_actions:
            if scenario == scenario_id:
                return action
        raise ScientificProtocolError(
            f"learned policy has no action for scenario {scenario_id!r}"
        )


@dataclass(frozen=True, slots=True)
class PairedBootstrapInterval:
    """Seed-level paired mean and deterministic bootstrap interval."""

    pair_count: int
    mean: float
    low: float
    high: float
    confidence_level: float = CONFIDENCE_LEVEL
    method: str = UNCERTAINTY_METHOD

    def content(self) -> dict[str, object]:
        return {
            "confidence_level": self.confidence_level,
            "high": self.high,
            "low": self.low,
            "mean": self.mean,
            "method": self.method,
            "pair_count": self.pair_count,
        }


@dataclass(frozen=True, slots=True)
class LocalLearnabilityCampaignResult:
    """Complete local qualification result and its explicit claim boundary."""

    manifest_sha256: str
    plan_sha256: str
    status: str
    planned_run_count: int
    receipts: tuple[EpisodeRunReceipt, ...]
    learned_policies: tuple[LearnedContextPolicy, ...]
    trained_return_mean: float | None
    trained_minus_frozen: PairedBootstrapInterval | None
    trained_minus_random: PairedBootstrapInterval | None
    quality_threshold: float
    quality_threshold_passed: bool
    deterministic_floor_separation_passed: bool
    learnability_witness_passed: bool
    source_identity_passed: bool
    source_profile_passed: bool
    all_planned_receipts_preserved: bool
    distinct_evaluation_config_count: int
    distinct_learned_policy_count: int
    uncertainty_interval_informative: bool
    scenario_generalization_passed: bool
    tuning_executed: bool
    external_prerequisite_evidence_bound: bool
    consumed_environment_steps: int
    environment_step_budget: int
    budget_respected: bool
    claim_grade: bool
    claim_limit: str = field(init=False, default=CLAIM_LIMIT)
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.claim_grade:
            raise ScientificProtocolError(
                "prototype anti-torpedo campaign cannot be claim-grade"
            )
        object.__setattr__(self, "content_sha256", _digest(self.content()))

    def content(self) -> dict[str, object]:
        return {
            "all_planned_receipts_preserved": self.all_planned_receipts_preserved,
            "claim_grade": self.claim_grade,
            "budget_respected": self.budget_respected,
            "consumed_environment_steps": self.consumed_environment_steps,
            "claim_limit": self.claim_limit,
            "deterministic_floor_separation_passed": (
                self.deterministic_floor_separation_passed
            ),
            "external_prerequisite_evidence_bound": (
                self.external_prerequisite_evidence_bound
            ),
            "environment_step_budget": self.environment_step_budget,
            "learnability_witness_passed": self.learnability_witness_passed,
            "learned_policies": [policy.content() for policy in self.learned_policies],
            "distinct_evaluation_config_count": (
                self.distinct_evaluation_config_count
            ),
            "distinct_learned_policy_count": self.distinct_learned_policy_count,
            "manifest_sha256": self.manifest_sha256,
            "plan_sha256": self.plan_sha256,
            "planned_run_count": self.planned_run_count,
            "quality_threshold": self.quality_threshold,
            "quality_threshold_passed": self.quality_threshold_passed,
            "receipts": [receipt.content() for receipt in self.receipts],
            "source_identity_passed": self.source_identity_passed,
            "source_profile_passed": self.source_profile_passed,
            "scenario_generalization_passed": (
                self.scenario_generalization_passed
            ),
            "status": self.status,
            "trained_minus_frozen": (
                None
                if self.trained_minus_frozen is None
                else self.trained_minus_frozen.content()
            ),
            "trained_minus_random": (
                None
                if self.trained_minus_random is None
                else self.trained_minus_random.content()
            ),
            "trained_return_mean": self.trained_return_mean,
            "tuning_executed": self.tuning_executed,
            "uncertainty_interval_informative": (
                self.uncertainty_interval_informative
            ),
        }


def _verify_manifest_identity(manifest: ScientificExperimentManifest) -> None:
    _verify_campaign_source_lock()
    expected = {
        "algorithm_config_sha256": ALGORITHM_CONFIG_SHA256,
        "environment_contract_sha256": ENVIRONMENT_CONTRACT_SHA256,
        "learner_implementation_sha256": LOADED_CAMPAIGN_IMPLEMENTATION_SHA256,
        "model_sha256": atsim_source_sha256(),
        "plugin_sha256": LOADED_ADAPTER_SOURCE_SHA256,
        "projection_contract_sha256": SEMANTIC_PROJECTION_CONTRACT_SHA256,
        "resource_budget_sha256": RESOURCE_BUDGET_SHA256,
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise ScientificProtocolError(
                f"anti-torpedo campaign identity mismatch: {name}"
            )
    if manifest.primary_endpoints != PRIMARY_ENDPOINTS:
        raise ScientificProtocolError("anti-torpedo primary endpoints differ")
    expected_training_budget = (
        len(manifest.measured_training_master_seeds)
        * len(ACTIONS)
        * len(SCENARIOS)
        * MAX_STEPS
    )
    expected_evaluation_episodes = (
        len(manifest.measured_training_master_seeds)
        * len(manifest.evaluation_seeds)
        * len(EVALUATION_POLICIES)
    )
    expected_total_budget = (
        expected_training_budget + expected_evaluation_episodes * MAX_STEPS
    )
    if manifest.evaluation_episodes != expected_evaluation_episodes:
        raise ScientificProtocolError(
            "evaluation_episodes must cover every master/seed/policy evaluation"
        )
    if manifest.evaluation_interval_steps != expected_training_budget:
        raise ScientificProtocolError(
            "evaluation_interval_steps must equal the frozen training budget"
        )
    if manifest.environment_step_budget != expected_total_budget:
        raise ScientificProtocolError(
            "environment_step_budget must equal the complete campaign step bound"
        )
    observed_scenarios = {
        scenario_for_seed(seed)[0] for seed in manifest.evaluation_seeds
    }
    if observed_scenarios != set(SCENARIOS):
        raise ScientificProtocolError(
            "evaluation seeds must hold out both anti-torpedo scenarios"
        )


def _training_plan(manifest: ScientificExperimentManifest) -> list[EpisodeRunPlan]:
    plans: list[EpisodeRunPlan] = []
    for master_seed in manifest.measured_training_master_seeds:
        ordinal = 0
        for scenario_id in SCENARIOS:
            for action in ACTIONS:
                while True:
                    episode_seed = derive_global_episode_seed(master_seed, ordinal)
                    selected_scenario = scenario_for_seed(episode_seed)[0]
                    candidate_ordinal = ordinal
                    ordinal += 1
                    if selected_scenario == scenario_id:
                        break
                    if ordinal > 100_000:
                        raise ScientificProtocolError(
                            "could not derive a scenario-balanced training plan"
                        )
                plans.append(
                    EpisodeRunPlan(
                        run_id=(
                            f"train:m{master_seed}:o{candidate_ordinal}:"
                            f"{scenario_id}:a{action}"
                        ),
                        phase="training",
                        master_seed=master_seed,
                        episode_seed=episode_seed,
                        global_episode_ordinal=candidate_ordinal,
                        scenario_id=scenario_id,
                        policy_id=f"explore-action-{action}",
                        fixed_action=action,
                    )
                )
    return plans


def _evaluation_plan(manifest: ScientificExperimentManifest) -> list[EpisodeRunPlan]:
    plans: list[EpisodeRunPlan] = []
    for master_seed in manifest.measured_training_master_seeds:
        for episode_seed in manifest.evaluation_seeds:
            scenario_id = scenario_for_seed(episode_seed)[0]
            for policy_id in EVALUATION_POLICIES:
                plans.append(
                    EpisodeRunPlan(
                        run_id=f"eval:m{master_seed}:s{episode_seed}:{policy_id}",
                        phase="evaluation",
                        master_seed=master_seed,
                        episode_seed=episode_seed,
                        global_episode_ordinal=None,
                        scenario_id=scenario_id,
                        policy_id=policy_id,
                        fixed_action=0 if policy_id == "frozen" else None,
                    )
                )
    return plans


def build_episode_plan(
    manifest: ScientificExperimentManifest,
) -> tuple[EpisodeRunPlan, ...]:
    """Return the complete deterministic run plan before execution."""

    training = _training_plan(manifest)
    evaluation = _evaluation_plan(manifest)
    training_episode_seeds = [item.episode_seed for item in training]
    if len(training_episode_seeds) != len(set(training_episode_seeds)):
        raise ScientificProtocolError(
            "derived measured-training episode seeds must be unique"
        )
    reserved = set(manifest.tuning_seeds).union(manifest.evaluation_seeds)
    overlap = sorted(set(training_episode_seeds).intersection(reserved))
    if overlap:
        raise ScientificProtocolError(
            "derived measured-training episode seeds overlap tuning/evaluation "
            f"seeds: {overlap}"
        )
    return tuple(training + evaluation)


def _random_floor_action(episode_seed: int, step: int) -> int:
    digest = hashlib.sha256(
        f"{RANDOM_FLOOR_ID}:{episode_seed}:{step}".encode("ascii")
    ).digest()
    return ACTIONS[int.from_bytes(digest[:4], "big") % len(ACTIONS)]


def _require_provenance_value(
    info: Mapping[str, object],
    key: str,
    expected: str,
) -> str:
    value = info.get(key)
    if value != expected:
        raise ScientificProtocolError(f"episode provenance mismatch: {key}")
    return expected


def _episode_provenance(
    info: Mapping[str, object],
    *,
    expected_atsim_model_source_sha256: str,
    expected_config_sha256: str,
    expected_policy_id: str | None = None,
) -> _EpisodeProvenance:
    policy_id = info.get("executor_qualification_policy_id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ScientificProtocolError(
            "episode provenance lacks executor qualification policy"
        )
    if expected_policy_id is not None and policy_id != expected_policy_id:
        raise ScientificProtocolError(
            "episode provenance mismatch: executor_qualification_policy_id"
        )
    source_claim_grade = info.get("claim_grade")
    executor_qualified = info.get("executor_qualified")
    if not isinstance(source_claim_grade, bool):
        raise ScientificProtocolError("episode provenance lacks claim_grade")
    if executor_qualified is not True:
        raise ScientificProtocolError(
            "episode executor is not admitted by its qualification policy"
        )
    return _EpisodeProvenance(
        atsim_model_source_sha256=_require_provenance_value(
            info,
            "atsim_model_source_sha256",
            expected_atsim_model_source_sha256,
        ),
        adapter_source_sha256=_require_provenance_value(
            info,
            "adapter_source_sha256",
            LOADED_ADAPTER_SOURCE_SHA256,
        ),
        environment_contract_sha256=_require_provenance_value(
            info,
            "environment_contract_sha256",
            ENVIRONMENT_CONTRACT_SHA256,
        ),
        projection_contract_sha256=_require_provenance_value(
            info,
            "projection_contract_sha256",
            SEMANTIC_PROJECTION_CONTRACT_SHA256,
        ),
        scenario_bank_sha256=_require_provenance_value(
            info,
            "scenario_bank_sha256",
            SCENARIO_BANK_SHA256,
        ),
        config_sha256=_require_provenance_value(
            info,
            "config_sha256",
            expected_config_sha256,
        ),
        pyjevsim_executor_profile_id=_require_provenance_value(
            info,
            "pyjevsim_executor_profile_id",
            PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id,
        ),
        pyjevsim_executor_revision=_require_provenance_value(
            info,
            "pyjevsim_executor_revision",
            PINNED_PYJEVSIM_2_1_2_PROFILE.revision,
        ),
        pyjevsim_executor_source_sha256=_require_provenance_value(
            info,
            "pyjevsim_executor_source_sha256",
            PYJEVSIM_EXECUTOR_SOURCE_SHA256,
        ),
        executor_qualification_policy_id=policy_id,
        source_claim_grade=source_claim_grade,
        executor_qualified=True,
    )


def _execute_episode(
    plan: EpisodeRunPlan,
    learned_actions: Mapping[str, int],
    *,
    expected_atsim_model_source_sha256: str,
) -> EpisodeRunReceipt:
    _scenario_id, scenario = scenario_for_seed(plan.episode_seed)
    expected_config_sha256 = scenario_config_sha256(scenario)
    environment = anti_torpedo_environment_factory(
        instance_id=plan.run_id,
        run_id="anti-torpedo-local-learnability",
        expected_atsim_model_source_sha256=expected_atsim_model_source_sha256,
        expected_adapter_source_sha256=LOADED_ADAPTER_SOURCE_SHA256,
        expected_environment_contract_sha256=ENVIRONMENT_CONTRACT_SHA256,
        expected_projection_contract_sha256=(
            SEMANTIC_PROJECTION_CONTRACT_SHA256
        ),
        expected_pyjevsim_executor_source_sha256=(
            PYJEVSIM_EXECUTOR_SOURCE_SHA256
        ),
        expected_scenario_bank_sha256=SCENARIO_BANK_SHA256,
    )
    actions: list[int] = []
    semantic_steps: list[object] = []
    total_return = 0.0
    outcome: str | None = None
    try:
        observation, reset_info = environment.reset(seed=plan.episode_seed)
        provenance = _episode_provenance(
            reset_info,
            expected_atsim_model_source_sha256=expected_atsim_model_source_sha256,
            expected_config_sha256=expected_config_sha256,
        )
        if reset_info.get("scenario_id") != plan.scenario_id:
            raise ScientificProtocolError("planned scenario differs at reset")
        semantic_steps.append(canonical_semantic_projection(observation))
        for step in range(MAX_STEPS):
            if plan.fixed_action is not None:
                action = plan.fixed_action
            elif plan.policy_id == "trained":
                action = learned_actions[plan.scenario_id]
            elif plan.policy_id == "random":
                action = _random_floor_action(plan.episode_seed, step)
            else:
                raise ScientificProtocolError(
                    f"episode plan has no executable policy: {plan.policy_id}"
                )
            actions.append(action)
            next_observation, reward, terminated, truncated, info = environment.step(
                action
            )
            _episode_provenance(
                info,
                expected_atsim_model_source_sha256=(
                    expected_atsim_model_source_sha256
                ),
                expected_config_sha256=expected_config_sha256,
                expected_policy_id=provenance.executor_qualification_policy_id,
            )
            total_return += float(reward)
            raw_outcome = info.get("outcome")
            outcome = raw_outcome if isinstance(raw_outcome, str) else None
            semantic_steps.append(
                canonical_semantic_projection(
                    {
                        "action": action,
                        "logical_time": info["logical_time"],
                        "observation": next_observation,
                        "outcome": outcome,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
            )
            if terminated or truncated:
                break
        if not math.isfinite(total_return):
            raise ScientificProtocolError("episode return is not finite")
        return EpisodeRunReceipt(
            plan=plan,
            status="completed",
            total_return=round(total_return, 12),
            steps=len(actions),
            outcome=outcome,
            action_trace_sha256=_digest(actions),
            semantic_sha256=semantic_projection_sha256(semantic_steps),
            atsim_model_source_sha256=provenance.atsim_model_source_sha256,
            adapter_source_sha256=provenance.adapter_source_sha256,
            environment_contract_sha256=provenance.environment_contract_sha256,
            projection_contract_sha256=provenance.projection_contract_sha256,
            scenario_bank_sha256=provenance.scenario_bank_sha256,
            config_sha256=provenance.config_sha256,
            pyjevsim_executor_profile_id=provenance.pyjevsim_executor_profile_id,
            pyjevsim_executor_revision=provenance.pyjevsim_executor_revision,
            pyjevsim_executor_source_sha256=(
                provenance.pyjevsim_executor_source_sha256
            ),
            executor_qualification_policy_id=(
                provenance.executor_qualification_policy_id
            ),
            source_claim_grade=provenance.source_claim_grade,
            executor_qualified=provenance.executor_qualified,
            error=None,
        )
    finally:
        environment.close()


EpisodeRunner = Callable[
    [EpisodeRunPlan, Mapping[str, int]], EpisodeRunReceipt
]


def _safe_execute(
    plan: EpisodeRunPlan,
    learned_actions: Mapping[str, int],
    runner: EpisodeRunner,
) -> EpisodeRunReceipt:
    try:
        return runner(plan, learned_actions)
    except Exception as exc:  # A terminal receipt must survive each planned failure.
        return EpisodeRunReceipt(
            plan=plan,
            status="failed",
            total_return=None,
            steps=0,
            outcome=None,
            action_trace_sha256=None,
            semantic_sha256=None,
            atsim_model_source_sha256=None,
            adapter_source_sha256=None,
            environment_contract_sha256=None,
            projection_contract_sha256=None,
            scenario_bank_sha256=None,
            config_sha256=None,
            pyjevsim_executor_profile_id=None,
            pyjevsim_executor_revision=None,
            pyjevsim_executor_source_sha256=None,
            executor_qualification_policy_id=None,
            source_claim_grade=False,
            executor_qualified=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def _learn_policy(
    master_seed: int,
    receipts: Sequence[EpisodeRunReceipt],
) -> LearnedContextPolicy | None:
    training = [
        receipt
        for receipt in receipts
        if receipt.plan.phase == "training"
        and receipt.plan.master_seed == master_seed
    ]
    if len(training) != len(ACTIONS) * len(SCENARIOS):
        return None
    values: dict[tuple[str, int], float] = {}
    for receipt in training:
        action = receipt.plan.fixed_action
        if (
            receipt.status != "completed"
            or receipt.total_return is None
            or action is None
        ):
            return None
        key = (receipt.plan.scenario_id, action)
        if key in values:
            raise ScientificProtocolError("duplicate contextual training sample")
        values[key] = receipt.total_return
    expected = {(scenario, action) for scenario in SCENARIOS for action in ACTIONS}
    if set(values) != expected:
        return None
    selected = tuple(
        (
            scenario,
            min(
                ACTIONS,
                key=lambda action: (-values[(scenario, action)], action),
            ),
        )
        for scenario in SCENARIOS
    )
    return LearnedContextPolicy(
        master_seed=master_seed,
        selected_actions=selected,
        action_returns=tuple(
            (scenario, action, values[(scenario, action)])
            for scenario in SCENARIOS
            for action in ACTIONS
        ),
        training_run_ids=tuple(receipt.plan.run_id for receipt in training),
    )


def _paired_bootstrap(
    differences: Sequence[float],
    *,
    manifest_sha256: str,
    comparison_id: str,
) -> PairedBootstrapInterval:
    if len(differences) < 10:
        raise ScientificProtocolError("paired inference requires at least 10 seeds")
    values = tuple(float(value) for value in differences)
    estimates: list[float] = []
    for sample in range(BOOTSTRAP_SAMPLES):
        selected: list[float] = []
        for draw in range(len(values)):
            digest = hashlib.sha256(
                (
                    f"{UNCERTAINTY_METHOD}:{manifest_sha256}:"
                    f"{comparison_id}:{sample}:{draw}"
                ).encode("ascii")
            ).digest()
            selected.append(values[int.from_bytes(digest[:8], "big") % len(values)])
        estimates.append(fmean(selected))
    estimates.sort()
    low_index = int(((1.0 - CONFIDENCE_LEVEL) / 2.0) * (len(estimates) - 1))
    high_index = int(
        ((1.0 + CONFIDENCE_LEVEL) / 2.0) * (len(estimates) - 1)
    )
    return PairedBootstrapInterval(
        pair_count=len(values),
        mean=round(fmean(values), 12),
        low=round(estimates[low_index], 12),
        high=round(estimates[high_index], 12),
    )


def _evaluation_statistics(
    manifest: ScientificExperimentManifest,
    receipts: Sequence[EpisodeRunReceipt],
) -> tuple[
    float | None,
    PairedBootstrapInterval | None,
    PairedBootstrapInterval | None,
]:
    trained_means: list[float] = []
    frozen_means: list[float] = []
    random_means: list[float] = []
    for master_seed in manifest.measured_training_master_seeds:
        by_policy: dict[str, list[float]] = {
            policy: [] for policy in EVALUATION_POLICIES
        }
        for receipt in receipts:
            if (
                receipt.plan.phase != "evaluation"
                or receipt.plan.master_seed != master_seed
            ):
                continue
            if receipt.status != "completed" or receipt.total_return is None:
                return None, None, None
            by_policy[receipt.plan.policy_id].append(receipt.total_return)
        if any(
            len(values) != len(manifest.evaluation_seeds)
            for values in by_policy.values()
        ):
            return None, None, None
        trained_means.append(fmean(by_policy["trained"]))
        frozen_means.append(fmean(by_policy["frozen"]))
        random_means.append(fmean(by_policy["random"]))
    frozen_difference = [
        trained - frozen
        for trained, frozen in zip(trained_means, frozen_means, strict=True)
    ]
    random_difference = [
        trained - random
        for trained, random in zip(trained_means, random_means, strict=True)
    ]
    return (
        round(fmean(trained_means), 12),
        _paired_bootstrap(
            frozen_difference,
            manifest_sha256=manifest.manifest_sha256,
            comparison_id="trained-minus-frozen",
        ),
        _paired_bootstrap(
            random_difference,
            manifest_sha256=manifest.manifest_sha256,
            comparison_id="trained-minus-random",
        ),
    )


def _floor_separation_passes(
    manifest: ScientificExperimentManifest,
    *,
    all_completed: bool,
    quality_passed: bool,
    frozen_interval: PairedBootstrapInterval | None,
    random_interval: PairedBootstrapInterval | None,
) -> bool:
    return (
        all_completed
        and quality_passed
        and frozen_interval is not None
        and random_interval is not None
        and frozen_interval.low > manifest.noninferiority_margin
        and random_interval.low > manifest.noninferiority_margin
    )


def _run_campaign(
    manifest: ScientificExperimentManifest,
    *,
    expected_manifest_sha256: str,
    runner: EpisodeRunner,
) -> LocalLearnabilityCampaignResult:
    manifest.verify(expected_manifest_sha256)
    _verify_manifest_identity(manifest)
    plan = build_episode_plan(manifest)
    plan_sha256 = _digest([item.content() for item in plan])
    training_receipts: list[EpisodeRunReceipt] = []
    for item in plan:
        if item.phase != "training":
            continue
        _verify_manifest_identity(manifest)
        training_receipts.append(_safe_execute(item, {}, runner))

    policies = tuple(
        policy
        for master_seed in manifest.measured_training_master_seeds
        if (
            policy := _learn_policy(master_seed, training_receipts)
        )
        is not None
    )
    policy_by_master = {policy.master_seed: policy for policy in policies}
    evaluation_receipts: list[EpisodeRunReceipt] = []
    for item in plan:
        if item.phase != "evaluation":
            continue
        _verify_manifest_identity(manifest)
        policy = policy_by_master.get(item.master_seed)
        if item.policy_id == "trained" and policy is None:
            evaluation_receipts.append(
                EpisodeRunReceipt(
                    plan=item,
                    status="blocked",
                    total_return=None,
                    steps=0,
                    outcome=None,
                    action_trace_sha256=None,
                    semantic_sha256=None,
                    atsim_model_source_sha256=None,
                    adapter_source_sha256=None,
                    environment_contract_sha256=None,
                    projection_contract_sha256=None,
                    scenario_bank_sha256=None,
                    config_sha256=None,
                    pyjevsim_executor_profile_id=None,
                    pyjevsim_executor_revision=None,
                    pyjevsim_executor_source_sha256=None,
                    executor_qualification_policy_id=None,
                    source_claim_grade=False,
                    executor_qualified=False,
                    error="training evidence incomplete",
                )
            )
            continue
        actions = (
            {}
            if policy is None
            else dict(policy.selected_actions)
        )
        evaluation_receipts.append(_safe_execute(item, actions, runner))

    receipts = tuple(training_receipts + evaluation_receipts)
    consumed_environment_steps = sum(receipt.steps for receipt in receipts)
    budget_respected = consumed_environment_steps <= manifest.environment_step_budget
    planned_ids = tuple(item.run_id for item in plan)
    receipt_ids = tuple(receipt.plan.run_id for receipt in receipts)
    all_receipts = planned_ids == receipt_ids
    source_identity_passed = all(
        receipt.status == "completed"
        and receipt.atsim_model_source_sha256 == manifest.model_sha256
        and receipt.adapter_source_sha256 == manifest.plugin_sha256
        and receipt.environment_contract_sha256
        == manifest.environment_contract_sha256
        and receipt.projection_contract_sha256
        == manifest.projection_contract_sha256
        and receipt.scenario_bank_sha256 == SCENARIO_BANK_SHA256
        and receipt.config_sha256
        == scenario_config_sha256(scenario_for_seed(receipt.plan.episode_seed)[1])
        and receipt.pyjevsim_executor_profile_id
        == PINNED_PYJEVSIM_2_1_2_PROFILE.profile_id
        and receipt.pyjevsim_executor_revision
        == PINNED_PYJEVSIM_2_1_2_PROFILE.revision
        and receipt.pyjevsim_executor_source_sha256
        == PYJEVSIM_EXECUTOR_SOURCE_SHA256
        and receipt.executor_qualification_policy_id is not None
        and receipt.executor_qualified
        for receipt in receipts
    )
    all_completed = (
        source_identity_passed
        and budget_respected
        and len(policies) == len(manifest.measured_training_master_seeds)
    )
    trained_mean, frozen_interval, random_interval = _evaluation_statistics(
        manifest, receipts
    )
    quality_passed = (
        trained_mean is not None and trained_mean >= manifest.quality_threshold
    )
    uncertainty_interval_informative = (
        frozen_interval is not None
        and random_interval is not None
        and frozen_interval.low < frozen_interval.high
        and random_interval.low < random_interval.high
    )
    deterministic_floor_separation_passed = _floor_separation_passes(
        manifest,
        all_completed=all_completed,
        quality_passed=quality_passed,
        frozen_interval=frozen_interval,
        random_interval=random_interval,
    )
    witness_passed = (
        deterministic_floor_separation_passed
        and uncertainty_interval_informative
    )
    # The adapter deliberately exposes a prototype source profile.  Keep this
    # closed even if a caller substitutes an episode runner in a unit test.
    source_profile_passed = (
        ANTI_TORPEDO_CLAIM_GRADE
        and source_identity_passed
        and all(receipt.source_claim_grade for receipt in receipts)
    )
    distinct_evaluation_config_count = len(
        {
            receipt.config_sha256
            for receipt in receipts
            if receipt.plan.phase == "evaluation"
            and receipt.status == "completed"
            and receipt.config_sha256 is not None
        }
    )
    distinct_learned_policy_count = len(
        {policy.selected_actions for policy in policies}
    )
    return LocalLearnabilityCampaignResult(
        manifest_sha256=manifest.manifest_sha256,
        plan_sha256=plan_sha256,
        status="completed" if all_completed else "failed",
        planned_run_count=len(plan),
        receipts=receipts,
        learned_policies=policies,
        trained_return_mean=trained_mean,
        trained_minus_frozen=frozen_interval,
        trained_minus_random=random_interval,
        quality_threshold=manifest.quality_threshold,
        quality_threshold_passed=quality_passed,
        deterministic_floor_separation_passed=(
            deterministic_floor_separation_passed
        ),
        learnability_witness_passed=witness_passed,
        source_identity_passed=source_identity_passed,
        source_profile_passed=source_profile_passed,
        all_planned_receipts_preserved=all_receipts,
        distinct_evaluation_config_count=distinct_evaluation_config_count,
        distinct_learned_policy_count=distinct_learned_policy_count,
        uncertainty_interval_informative=uncertainty_interval_informative,
        scenario_generalization_passed=False,
        tuning_executed=False,
        external_prerequisite_evidence_bound=False,
        consumed_environment_steps=consumed_environment_steps,
        environment_step_budget=manifest.environment_step_budget,
        budget_respected=budget_respected,
        claim_grade=False,
    )


def run_local_learnability_campaign(
    manifest: ScientificExperimentManifest,
    *,
    expected_manifest_sha256: str,
) -> LocalLearnabilityCampaignResult:
    """Execute the complete local campaign after fail-closed manifest checks."""

    def source_locked_runner(
        plan: EpisodeRunPlan,
        learned_actions: Mapping[str, int],
    ) -> EpisodeRunReceipt:
        return _execute_episode(
            plan,
            learned_actions,
            expected_atsim_model_source_sha256=manifest.model_sha256,
        )

    return _run_campaign(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
        runner=source_locked_runner,
    )
