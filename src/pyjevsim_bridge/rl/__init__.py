"""Reinforcement-learning interfaces for pyjevsim and gorti.

The core is dependency-free and Gymnasium-shaped.  Model authors provide an
``EpisodeFactory`` that builds a fresh pyjevsim ``SysExecutor`` graph on every
reset; the same environment can then be hosted by ``LocalRolloutPool`` or a
``GortiRolloutChannel``.
"""

from importlib import import_module

from pyjevsim_bridge.rl.adapters import FunctionalEpisodeBinding
from pyjevsim_bridge.rl.batching import (
    AdaptiveBatchPolicy,
    AdaptiveTransitionBatcher,
    BatchFlushReason,
    BatchingTelemetry,
    PlannedTransitionBatch,
)
from pyjevsim_bridge.rl.contracts import (
    ArraySnapshot,
    EnumSnapshot,
    EnvironmentClosedError,
    EpisodeBinding,
    EpisodeCleanupError,
    EpisodeContext,
    EpisodeFactory,
    EpisodeLifecycleError,
    EpisodeStateError,
    EpisodeStepError,
    ExecutorContractError,
    ExecutorProtocol,
    RLEnvironmentError,
    StepView,
    StepViewSnapshotError,
)
from pyjevsim_bridge.rl.environment import PyJevSimEnv
from pyjevsim_bridge.rl.executor import (
    PINNED_PYJEVSIM_2_1_2_PROFILE,
    BindingDecisionBoundary,
    DecisionBoundary,
    ExecutorDriver,
    ExecutorQualificationPolicy,
    ExecutorQualificationProfile,
    ExecutorSemanticCapability,
    ExecutorSemanticEvidence,
    ExecutorSemanticRequirement,
    ExecutorStep,
    FixedDeltaBoundary,
    NextEventBoundary,
)
from pyjevsim_bridge.rl.federation import (
    ACTION_CLASS,
    CONTROL_CLASS,
    POLICY_CLASS,
    RECEIPT_CLASS,
    TRANSITION_CLASS,
    EnvelopeValidationError,
    EventStreamExhaustedError,
    FederationHaltedError,
    FederationProtocolError,
    GortiRolloutChannel,
    GrantedBatch,
    IdempotencyConflictError,
    ReceivedEnvelope,
    Role,
    canonical_json,
    decode_envelope,
    encode_envelope,
    validate_envelope,
)
from pyjevsim_bridge.rl.learning import (
    EMPTY_TRANSITION_BATCH_SHA256,
    POLICY_ARTIFACT_SCHEMA_VERSION,
    POLICY_REGISTRY_EPISODE_ID,
    TABULAR_Q_ALGORITHM_ID,
    TABULAR_Q_ALGORITHM_VERSION,
    TABULAR_Q_MEDIA_TYPE,
    ActorPolicy,
    InMemoryPolicyArtifactStore,
    LearnerAdapter,
    LearnerRecoveryState,
    LearnerSession,
    LearningContractError,
    LoadedPolicy,
    PolicyActionBatch,
    PolicyAnnouncement,
    PolicyArtifact,
    PolicyArtifactError,
    PolicyArtifactRef,
    PolicyArtifactStore,
    PolicyCandidate,
    PolicyCompatibility,
    PolicyCompatibilityError,
    PolicyIntegrityError,
    PolicyLoader,
    PolicyVersionError,
    TabularPolicyLoader,
    TabularQLearnerAdapter,
    TransitionBatchValidationError,
    ValidatedTransitionBatch,
)
from pyjevsim_bridge.rl.local import (
    LocalRolloutBatchError,
    LocalRolloutCloseError,
    LocalRolloutPool,
    LocalRolloutPoolState,
    LocalRolloutRecoveryError,
    WorkerCloseReceipt,
    derive_episode_seed,
)
from pyjevsim_bridge.rl.local_process import (
    LocalProcessWorkerError,
    ProcessBackendOptions,
    ProcessCleanupReceipt,
)
from pyjevsim_bridge.rl.records import ActionCommand, ResetResult, TransitionRecord
from pyjevsim_bridge.rl.scientific import (
    DEFAULT_PROJECTION_EXCLUDED_KEYS,
    GLOBAL_EPISODE_SEED_DERIVATION_ID,
    PROJECTION_CONTRACT_ID,
    SCIENTIFIC_MANIFEST_SCHEMA_VERSION,
    SEMANTIC_PROJECTION_CONTRACT_SHA256,
    ScientificExperimentManifest,
    ScientificProtocolError,
    canonical_semantic_projection,
    derive_global_episode_seed,
    semantic_projection_contract_sha256,
    semantic_projection_sha256,
)
from pyjevsim_bridge.rl.scientific_matched import (
    AnalysisReceipt,
    ArtifactReference,
    CapabilityDisposition,
    CapabilityKind,
    CapabilityReceipt,
    CellSpec,
    ClaimDecision,
    ClaimKind,
    ClaimMatrix,
    EpisodeReceipt,
    EvidenceLedger,
    ExecutionTimeline,
    ExternalRegistrationReceipt,
    JoinedCapabilityReceipt,
    JoinedPhaseArtifact,
    LearnerEngine,
    MatchedMeasuredManifestV1,
    MatchedProtocolError,
    MatchedProtocolManifestV1,
    MeasuredSessionPlan,
    RegistrationRequirement,
    RegistrationStage,
    ResourceCounters,
    RolloutPlane,
    RunStatus,
    ScenarioCase,
    ScenarioPartition,
    SessionReceipt,
    SourceLock,
    TuningCompletionReceipt,
    bootstrap_resample_seed,
    build_claim_matrix,
    build_measured_plan,
    cell_capability_sha256,
    measured_plan_sha256,
    read_evidence_ledger,
    validate_evaluation_episode_receipts,
    validate_external_registrations,
    validate_joined_capability,
    validate_measured_plan,
    validate_native_capability,
    validate_terminal_receipts,
    verify_indexed_artifacts,
    verify_ledger_artifacts,
    write_evidence_ledger,
)

_REFERENCE_PPO_LAZY_EXPORTS = frozenset(
    {
        "FROZEN_REFERENCE_PPO_CONFIG_SHA256",
        "LOADED_REFERENCE_PPO_SOURCE_SHA256",
        "PPOInferenceInputV1",
        "REFERENCE_PPO_ACTION_SCHEMA_SHA256",
        "ReferencePPOCapabilityReceipt",
        "ReferencePPOCapabilityReceiptV1",
        "ReferencePPOCheckpointV1",
        "ReferencePPOConfigV1",
        "ReferencePPOLearnerAdapter",
        "ReferencePPOPolicyLoader",
        "compute_gae",
        "numpy_runtime_identity",
        "numpy_runtime_sha256",
    }
)
_ANTI_TORPEDO_FEATURE_LAZY_EXPORTS = frozenset(
    {
        "ANTI_TORPEDO_ACTION_COUNT",
        "ANTI_TORPEDO_FEATURE_CONTRACT_SHA256",
        "ANTI_TORPEDO_FEATURE_SCHEMA_VERSION",
        "ANTI_TORPEDO_FEATURE_SIZE",
        "PPO_INFERENCE_INPUT_SCHEMA_VERSION",
        "AntiTorpedoFeatureError",
        "AntiTorpedoV2FeatureContract",
    }
)
_REFERENCE_PPO_QUALIFICATION_LAZY_EXPORTS = frozenset(
    {
        "QualificationError",
        "ReferencePPOQualificationBundle",
        "ReferencePPOQualificationPlan",
        "VerifiedReferencePPOQualificationReceipt",
        "run_reference_ppo_qualification",
        "verify_qualification_bundle",
    }
)
_JOINED_REFERENCE_PPO_LAZY_EXPORTS = frozenset(
    {
        "AssignmentReceipt",
        "EpisodeFederationTimeMapper",
        "FilesystemPolicyArtifactStore",
        "JoinedAdmissionError",
        "JoinedAdmissionReceipt",
        "JoinedEvaluationAssignmentV1",
        "JoinedPhase",
        "JoinedPhaseCutV1",
        "JoinedReferenceCoordinatorRuntime",
        "JoinedReferenceLaunchPlanV1",
        "JoinedReferenceWorkerRuntime",
        "PHASE_BASES",
        "PolicyActivationReceipt",
        "TIME_STRIDE",
        "TerminalReceipt",
        "TimeGrantReceipt",
        "assignment_set_sha256",
        "validate_joined_admission",
        "verify_joined_reference_process_results",
    }
)
_JOINED_REFERENCE_PPO_QUALIFICATION_LAZY_EXPORTS = frozenset(
    {
        "CAPABILITY_SCOPE",
        "JoinedArtifactEntry",
        "JoinedQualificationError",
        "JoinedReferencePPOBundle",
        "JoinedReferencePPOCapability",
        "JoinedReferencePPOExecution",
        "JoinedReferencePPOPlan",
        "JoinedReferencePPORunner",
        "JoinedVerificationContext",
        "run_joined_reference_ppo_qualification",
        "verify_joined_reference_ppo_bundle",
    }
)


def __getattr__(name: str) -> object:
    """Load the optional NumPy/PyJevSim reference stack only when requested."""

    if name in _REFERENCE_PPO_LAZY_EXPORTS:
        module_name = "pyjevsim_bridge.rl.reference_ppo"
    elif name in _ANTI_TORPEDO_FEATURE_LAZY_EXPORTS:
        module_name = (
            "pyjevsim_bridge.rl.qualification_models.anti_torpedo_features"
        )
    elif name in _REFERENCE_PPO_QUALIFICATION_LAZY_EXPORTS:
        module_name = "pyjevsim_bridge.rl.reference_ppo_qualification"
    elif name in _JOINED_REFERENCE_PPO_LAZY_EXPORTS:
        module_name = "pyjevsim_bridge.rl.joined_reference_ppo"
    elif name in _JOINED_REFERENCE_PPO_QUALIFICATION_LAZY_EXPORTS:
        module_name = "pyjevsim_bridge.rl.joined_reference_ppo_qualification"
    else:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def build_controlled_counter_episode(context: EpisodeContext) -> EpisodeBinding:
    """Lazily load the optional real-PyJevSim controlled model factory."""

    from pyjevsim_bridge.rl.qualification_models import build_episode

    return build_episode(context)


def build_anti_torpedo_episode(context: EpisodeContext) -> EpisodeBinding:
    """Lazily load the nontrivial external-source AT/SIM episode factory."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
        build_anti_torpedo_episode as implementation,
    )

    return implementation(context)


def anti_torpedo_environment_factory(
    *,
    instance_id: str = "env-0",
    run_id: str = "local",
    expected_atsim_model_source_sha256: str | None = None,
    expected_adapter_source_sha256: str | None = None,
    expected_environment_contract_sha256: str | None = None,
    expected_projection_contract_sha256: str | None = None,
    expected_pyjevsim_executor_source_sha256: str | None = None,
    expected_scenario_bank_sha256: str | None = None,
) -> PyJevSimEnv:
    """Lazily construct the default prototype AT/SIM environment."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
        anti_torpedo_environment_factory as implementation,
    )

    return implementation(
        instance_id=instance_id,
        run_id=run_id,
        expected_atsim_model_source_sha256=expected_atsim_model_source_sha256,
        expected_adapter_source_sha256=expected_adapter_source_sha256,
        expected_environment_contract_sha256=(
            expected_environment_contract_sha256
        ),
        expected_projection_contract_sha256=expected_projection_contract_sha256,
        expected_pyjevsim_executor_source_sha256=(
            expected_pyjevsim_executor_source_sha256
        ),
        expected_scenario_bank_sha256=expected_scenario_bank_sha256,
    )


def anti_torpedo_v2_environment_factory(
    *,
    instance_id: str = "env-v2-0",
    run_id: str = "local-v2",
    expected_atsim_model_source_sha256: str | None = None,
    expected_adapter_source_sha256: str | None = None,
    expected_environment_contract_sha256: str | None = None,
    expected_projection_contract_sha256: str | None = None,
    expected_pyjevsim_executor_source_sha256: str | None = None,
    expected_profile_generator_source_sha256: str | None = None,
    expected_scenario_bank_sha256: str | None = None,
    expected_scenario_family_sha256: str | None = None,
    expected_scenario_source_sha256: str | None = None,
    expected_factor_schema_sha256: str | None = None,
) -> PyJevSimEnv:
    """Lazily construct the explicit-ordinal effective-workload environment."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
        anti_torpedo_v2_environment_factory as implementation,
    )

    return implementation(
        instance_id=instance_id,
        run_id=run_id,
        expected_atsim_model_source_sha256=expected_atsim_model_source_sha256,
        expected_adapter_source_sha256=expected_adapter_source_sha256,
        expected_environment_contract_sha256=expected_environment_contract_sha256,
        expected_projection_contract_sha256=expected_projection_contract_sha256,
        expected_pyjevsim_executor_source_sha256=(
            expected_pyjevsim_executor_source_sha256
        ),
        expected_profile_generator_source_sha256=(
            expected_profile_generator_source_sha256
        ),
        expected_scenario_bank_sha256=expected_scenario_bank_sha256,
        expected_scenario_family_sha256=expected_scenario_family_sha256,
        expected_scenario_source_sha256=expected_scenario_source_sha256,
        expected_factor_schema_sha256=expected_factor_schema_sha256,
    )


def build_anti_torpedo_campaign_manifest() -> ScientificExperimentManifest:
    """Build the source-locked bounded anti-torpedo campaign manifest."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo_campaign import (
        build_local_campaign_manifest,
    )

    return build_local_campaign_manifest()


def run_anti_torpedo_local_learnability_campaign(
    manifest: ScientificExperimentManifest,
    *,
    expected_manifest_sha256: str,
) -> object:
    """Run the local campaign while retaining its non-claim-grade boundary."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo_campaign import (
        run_local_learnability_campaign,
    )

    return run_local_learnability_campaign(
        manifest,
        expected_manifest_sha256=expected_manifest_sha256,
    )

__all__ = [
    "ACTION_CLASS",
    "CONTROL_CLASS",
    "DEFAULT_PROJECTION_EXCLUDED_KEYS",
    "EMPTY_TRANSITION_BATCH_SHA256",
    "POLICY_CLASS",
    "RECEIPT_CLASS",
    "POLICY_ARTIFACT_SCHEMA_VERSION",
    "PINNED_PYJEVSIM_2_1_2_PROFILE",
    "POLICY_REGISTRY_EPISODE_ID",
    "PROJECTION_CONTRACT_ID",
    "TABULAR_Q_ALGORITHM_ID",
    "TABULAR_Q_ALGORITHM_VERSION",
    "TABULAR_Q_MEDIA_TYPE",
    "TRANSITION_CLASS",
    "ActionCommand",
    "ActorPolicy",
    "AdaptiveBatchPolicy",
    "AdaptiveTransitionBatcher",
    "AnalysisReceipt",
    "ArraySnapshot",
    "ArtifactReference",
    "BindingDecisionBoundary",
    "BatchFlushReason",
    "BatchingTelemetry",
    "CapabilityDisposition",
    "CapabilityKind",
    "CapabilityReceipt",
    "CellSpec",
    "ClaimDecision",
    "ClaimKind",
    "ClaimMatrix",
    "DecisionBoundary",
    "EnvelopeValidationError",
    "EnvironmentClosedError",
    "EnumSnapshot",
    "EpisodeBinding",
    "EpisodeCleanupError",
    "EpisodeContext",
    "EpisodeFactory",
    "EpisodeLifecycleError",
    "EpisodeReceipt",
    "EpisodeStateError",
    "EpisodeStepError",
    "EventStreamExhaustedError",
    "ExecutorContractError",
    "ExecutorDriver",
    "ExecutorQualificationPolicy",
    "ExecutorQualificationProfile",
    "ExecutorProtocol",
    "ExecutorSemanticCapability",
    "ExecutorSemanticEvidence",
    "ExecutorSemanticRequirement",
    "ExecutorStep",
    "EvidenceLedger",
    "ExecutionTimeline",
    "ExternalRegistrationReceipt",
    "FederationHaltedError",
    "FederationProtocolError",
    "FixedDeltaBoundary",
    "FunctionalEpisodeBinding",
    "GortiRolloutChannel",
    "GLOBAL_EPISODE_SEED_DERIVATION_ID",
    "GrantedBatch",
    "IdempotencyConflictError",
    "InMemoryPolicyArtifactStore",
    "JoinedCapabilityReceipt",
    "JoinedPhaseArtifact",
    "LearnerAdapter",
    "LearnerRecoveryState",
    "LearnerEngine",
    "LearnerSession",
    "LearningContractError",
    "LoadedPolicy",
    "LocalRolloutBatchError",
    "LocalRolloutCloseError",
    "LocalRolloutPool",
    "LocalRolloutPoolState",
    "LocalRolloutRecoveryError",
    "MatchedMeasuredManifestV1",
    "MatchedProtocolError",
    "MatchedProtocolManifestV1",
    "MeasuredSessionPlan",
    "LocalProcessWorkerError",
    "NextEventBoundary",
    "PyJevSimEnv",
    "ProcessBackendOptions",
    "ProcessCleanupReceipt",
    "PlannedTransitionBatch",
    "PolicyActionBatch",
    "PolicyAnnouncement",
    "PolicyArtifact",
    "PolicyArtifactError",
    "PolicyArtifactRef",
    "PolicyArtifactStore",
    "PolicyCandidate",
    "PolicyCompatibility",
    "PolicyCompatibilityError",
    "PolicyIntegrityError",
    "PolicyLoader",
    "PolicyVersionError",
    "RLEnvironmentError",
    "ReceivedEnvelope",
    "RegistrationRequirement",
    "RegistrationStage",
    "ResetResult",
    "ResourceCounters",
    "Role",
    "RolloutPlane",
    "RunStatus",
    "SCIENTIFIC_MANIFEST_SCHEMA_VERSION",
    "SEMANTIC_PROJECTION_CONTRACT_SHA256",
    "ScientificExperimentManifest",
    "ScientificProtocolError",
    "ScenarioCase",
    "ScenarioPartition",
    "SessionReceipt",
    "SourceLock",
    "StepView",
    "StepViewSnapshotError",
    "TabularPolicyLoader",
    "TabularQLearnerAdapter",
    "TransitionRecord",
    "TuningCompletionReceipt",
    "TransitionBatchValidationError",
    "ValidatedTransitionBatch",
    "WorkerCloseReceipt",
    "canonical_json",
    "canonical_semantic_projection",
    "build_controlled_counter_episode",
    "build_claim_matrix",
    "build_measured_plan",
    "build_anti_torpedo_campaign_manifest",
    "build_anti_torpedo_episode",
    "decode_envelope",
    "bootstrap_resample_seed",
    "cell_capability_sha256",
    "derive_episode_seed",
    "derive_global_episode_seed",
    "encode_envelope",
    "anti_torpedo_environment_factory",
    "anti_torpedo_v2_environment_factory",
    "run_anti_torpedo_local_learnability_campaign",
    "measured_plan_sha256",
    "read_evidence_ledger",
    "semantic_projection_sha256",
    "semantic_projection_contract_sha256",
    "validate_evaluation_episode_receipts",
    "validate_external_registrations",
    "validate_joined_capability",
    "validate_measured_plan",
    "validate_native_capability",
    "validate_terminal_receipts",
    "validate_envelope",
    "verify_indexed_artifacts",
    "verify_ledger_artifacts",
    "write_evidence_ledger",
    # Joined reference PPO federation and qualification stack.
    "CAPABILITY_SCOPE",
    "PHASE_BASES",
    "TIME_STRIDE",
    "AssignmentReceipt",
    "EpisodeFederationTimeMapper",
    "FilesystemPolicyArtifactStore",
    "JoinedAdmissionError",
    "JoinedAdmissionReceipt",
    "JoinedArtifactEntry",
    "JoinedEvaluationAssignmentV1",
    "JoinedPhase",
    "JoinedPhaseCutV1",
    "JoinedQualificationError",
    "JoinedReferenceCoordinatorRuntime",
    "JoinedReferenceLaunchPlanV1",
    "JoinedReferencePPOBundle",
    "JoinedReferencePPOCapability",
    "JoinedReferencePPOExecution",
    "JoinedReferencePPOPlan",
    "JoinedReferencePPORunner",
    "JoinedReferenceWorkerRuntime",
    "JoinedVerificationContext",
    "PolicyActivationReceipt",
    "TerminalReceipt",
    "TimeGrantReceipt",
    "assignment_set_sha256",
    "run_joined_reference_ppo_qualification",
    "validate_joined_admission",
    "verify_joined_reference_ppo_bundle",
    "verify_joined_reference_process_results",
    # Optional NumPy/PyJevSim reference stack, resolved by __getattr__.
    "ANTI_TORPEDO_ACTION_COUNT",
    "ANTI_TORPEDO_FEATURE_CONTRACT_SHA256",
    "ANTI_TORPEDO_FEATURE_SCHEMA_VERSION",
    "ANTI_TORPEDO_FEATURE_SIZE",
    "FROZEN_REFERENCE_PPO_CONFIG_SHA256",
    "LOADED_REFERENCE_PPO_SOURCE_SHA256",
    "PPOInferenceInputV1",
    "PPO_INFERENCE_INPUT_SCHEMA_VERSION",
    "QualificationError",
    "REFERENCE_PPO_ACTION_SCHEMA_SHA256",
    "AntiTorpedoFeatureError",
    "AntiTorpedoV2FeatureContract",
    "ReferencePPOCapabilityReceipt",
    "ReferencePPOCapabilityReceiptV1",
    "ReferencePPOCheckpointV1",
    "ReferencePPOConfigV1",
    "ReferencePPOLearnerAdapter",
    "ReferencePPOPolicyLoader",
    "ReferencePPOQualificationBundle",
    "ReferencePPOQualificationPlan",
    "VerifiedReferencePPOQualificationReceipt",
    "compute_gae",
    "numpy_runtime_identity",
    "numpy_runtime_sha256",
    "run_reference_ppo_qualification",
    "verify_qualification_bundle",
]
