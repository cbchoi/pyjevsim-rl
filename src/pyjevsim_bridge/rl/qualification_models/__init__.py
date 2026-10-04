"""Repository-controlled real-PyJevSim qualification models."""

from pyjevsim_bridge.rl.contracts import EpisodeBinding, EpisodeContext


def __getattr__(name: str) -> object:
    """Keep model classes lazy so transport-neutral RL imports stay dependency-free."""

    if name == "ControlledCounter":
        from pyjevsim_bridge.rl.qualification_models.controlled_counter import (
            ControlledCounter,
        )

        globals()[name] = ControlledCounter
        return ControlledCounter
    raise AttributeError(name)


def build_episode(context: EpisodeContext) -> EpisodeBinding:
    """Lazily import the real-PyJevSim controlled-counter factory."""

    from pyjevsim_bridge.rl.qualification_models.controlled_counter import (
        build_episode as implementation,
    )

    return implementation(context)


def build_anti_torpedo_episode(context: object) -> object:
    """Lazily import the external-source AT/SIM workload adapter."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo import (
        build_anti_torpedo_episode as implementation,
    )

    return implementation(context)  # type: ignore[arg-type]


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
) -> object:
    """Lazily construct the default AT/SIM workload environment."""

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
) -> object:
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


def build_anti_torpedo_campaign_manifest() -> object:
    """Lazily build the source-locked local learnability manifest."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo_campaign import (
        build_local_campaign_manifest,
    )

    return build_local_campaign_manifest()


def run_anti_torpedo_local_learnability_campaign(
    manifest: object,
    *,
    expected_manifest_sha256: str,
) -> object:
    """Lazily execute the bounded local anti-torpedo learning campaign."""

    from pyjevsim_bridge.rl.qualification_models.anti_torpedo_campaign import (
        run_local_learnability_campaign,
    )

    return run_local_learnability_campaign(
        manifest,  # type: ignore[arg-type]
        expected_manifest_sha256=expected_manifest_sha256,
    )

__all__ = [
    "ControlledCounter",
    "anti_torpedo_environment_factory",
    "anti_torpedo_v2_environment_factory",
    "build_anti_torpedo_campaign_manifest",
    "build_anti_torpedo_episode",
    "build_episode",
    "run_anti_torpedo_local_learnability_campaign",
]
