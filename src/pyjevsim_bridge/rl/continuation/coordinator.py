"""Opt-in continuation transactions over explicitly installed state owners.

Providers are trusted implementation code, not a sandbox. Round-trip checks do
not replace independent conformance tests or a model's state-closure argument.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from .contracts import (
    SNAPSHOT_SCHEMA, BranchContext, CapabilityReport, CaptureRequest,
    CleanupReceipt, ContinuationError, ContinuationSnapshot,
    ResetRequest, RuntimeParts, Violation, canonical_bytes, digest,
    freeze_value, thaw_value,
)
from .envelope import decode_checked, encode_snapshot
from .lifecycle import CleanupLedger
from .references import ReferenceRegistry
from .registry import ContinuationRegistry


class RuntimeHandle:
    """Only supported mutation entry point; do not mutate exposed internals."""

    def __init__(self, coordinator, bundle, parts, cleanup):
        self._coordinator = coordinator
        self._bundle = bundle
        self._parts = parts
        self._cleanup = cleanup
        self._lock = threading.RLock()
        self._state = 'READY'
        self._cleanup_errors: tuple[str, ...] = ()
        self._execution_profile = coordinator.execution_profile
        self._admission_witness = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def profile_id(self) -> str:
        return self._bundle.profile.profile_id

    @property
    def execution_profile(self) -> str:
        """Execution assurance contract, separate from snapshot semantics."""
        return self._execution_profile

    def step(self, action):
        with self._coordinator._locked(self):
            self._coordinator._require_ready(self)
            self._coordinator._admit_step(self)
            self._state = 'STEPPING'
            try:
                result = self._parts.env.step(action)
                self._state = 'DONE' if result[2] or result[3] else 'READY'
                return result
            except BaseException:
                self._state = 'INVALID'
                self._admission_witness = None
                raise

    def close(self) -> CleanupReceipt:
        with self._coordinator._locked(self, allow_closed=True):
            if self._state in ('STEPPING', 'CAPTURING', 'CLOSING'):
                raise ContinuationError('CC_BUSY', 'operation in progress')
            self._state = 'CLOSING'
            self._admission_witness = None
            result = self._cleanup.close()
            self._cleanup_errors += tuple(result.errors)
            self._state = 'CLOSED' if result.success else 'INVALID'
            # A later successful release does not erase a previous failure.
            return CleanupReceipt(result.success and not self._cleanup_errors,
                                  result.failed_resources, self._cleanup_errors)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        receipt = self.close()
        if not receipt.success:
            raise ContinuationError('CC_CLEANUP_UNCONFIRMED', 'owned cleanup failed',
                                    cleanup_errors=receipt.errors) from exc


class ContinuationCoordinator:
    """Continuation transactions with explicit ordinary-step assurance.

    ``strict-v1`` preserves full admission before every step.
    ``admitted-runtime-v1`` is opt-in trusted-model execution: installed source,
    providers, callbacks and validators must not change during a handle's life;
    callers must not mutate engine/model/environment internals. Supported model
    transitions must preserve their declared invariants. Ordinary steps retain
    lifetime/ownership checks and the unchanged environment execution path, but
    do not immediately detect arbitrary internal or installation mutation.
    Capture, restore, inspection and semantic views remain full checkpoints.
    Neither profile is a sandbox against arbitrary Python modification.
    """

    def __init__(self, registry: ContinuationRegistry, *, execution_profile="strict-v1"):
        if type(execution_profile) is not str or execution_profile not in (
                "strict-v1", "admitted-runtime-v1"):
            raise ContinuationError('CC_UNSUPPORTED_PROFILE', 'unknown execution profile')
        self.registry = registry
        self._execution_profile = execution_profile

    @property
    def execution_profile(self) -> str:
        return self._execution_profile

    def _publish_runtime(self, bundle, parts, cleanup):
        # The callers have completed the full fresh/restore validation. Mint no
        # witness for strict execution and add no source reads to its path.
        runtime = RuntimeHandle(self, bundle, parts, cleanup)
        if self.execution_profile == 'admitted-runtime-v1':
            runtime._admission_witness = self.registry._admitted_runtime_witness(bundle, runtime)
        return runtime

    def _admit_step(self, runtime):
        if runtime.execution_profile == 'strict-v1':
            self._admit(runtime._bundle, runtime._parts)
        elif runtime.execution_profile == 'admitted-runtime-v1':
            self.registry._require_admitted_runtime(runtime._admission_witness,
                bundle=runtime._bundle, runtime=runtime)
        else:
            raise ContinuationError('CC_UNSUPPORTED_PROFILE', 'unknown runtime execution profile')

    @contextmanager
    def _locked(self, runtime, *, allow_closed=False):
        if type(runtime) is not RuntimeHandle or runtime._coordinator is not self:
            raise ContinuationError('CC_INCOMPATIBLE_IDENTITY', 'foreign runtime handle')
        if not runtime._lock.acquire(blocking=False):
            raise ContinuationError('CC_BUSY', 'runtime operation lock unavailable')
        env_lock = getattr(runtime._parts.env, '_lifecycle_lock', None)
        acquired = False
        try:
            if env_lock is None or not env_lock.acquire(blocking=False):
                raise ContinuationError('CC_BUSY', 'environment lifecycle lock unavailable')
            acquired = True
            env = runtime._parts.env
            if any(getattr(env, field, False) for field in
                   ('_stepping', '_resetting', '_closing', '_constructing', '_disposing')):
                raise ContinuationError('CC_BUSY', 'environment callback/lifecycle active')
            if not allow_closed and runtime._state == 'CLOSED':
                raise ContinuationError('CC_INVALID_BOUNDARY', 'runtime closed')
            yield
        finally:
            if acquired:
                env_lock.release()
            runtime._lock.release()

    @staticmethod
    def _require_ready(runtime):
        if runtime._state in ('STEPPING', 'CAPTURING', 'CLOSING'):
            raise ContinuationError('CC_BUSY', 'runtime operation active')
        if runtime._state != 'READY':
            raise ContinuationError('CC_INVALID_BOUNDARY', f'runtime is {runtime._state}')

    @staticmethod
    def _cleanup_failure(cleanup, original, phase):
        result = cleanup.close()
        code = 'CC_RESTORE_FAILED' if result.success else 'CC_CLEANUP_UNCONFIRMED'
        return ContinuationError(code, str(original), phase=phase,
                                 state_disposition='candidate-not-published',
                                 cleanup_errors=result.errors)

    def _states(self, bundle, parts):
        return {
            'engine_state': bundle.engine.export_state(parts, parts.refs),
            'model_state': bundle.model.export_state(parts.graph, parts.refs),
            'boundary_state': bundle.boundary.export_state(parts),
        }

    def _violations(self, bundle, parts):
        topology = bundle.model.topology()
        reasons = list(bundle.engine.inspect(parts, topology))
        if reasons:
            return tuple(reasons)
        reasons.extend(bundle.model.inspect(parts.graph))
        if reasons:
            return tuple(reasons)
        engine_state = bundle.engine.export_state(parts, parts.refs)
        reasons.extend(bundle.boundary.inspect(parts, engine_state))
        return tuple(reasons)

    def _admit(self, bundle, parts):
        self.registry.resolve(bundle.profile)
        reasons = self._violations(bundle, parts)
        if reasons:
            raise ContinuationError(reasons[0].code, reasons[0].message,
                                    phase='admission')

    def _payload(self, bundle, parts, logical, *, states=None):
        profile = bundle.profile
        return {
            'schema_id': SNAPSHOT_SCHEMA,
            'profile': profile.to_payload(),
            'identities': {
                'engine_source_sha256': profile.engine.implementation_sha256,
                'model_source_sha256': profile.model.implementation_sha256,
                'boundary_source_sha256': profile.boundary.implementation_sha256,
                'runtime_sha256': profile.runtime_sha256,
                'config_sha256': parts.metadata['config_sha256'],
                'obligation_manifest_sha256': profile.obligation_manifest_sha256,
            },
            'logical_context': thaw_value(logical),
            'topology': bundle.model.topology(),
            **(self._states(bundle, parts) if states is None else states),
        }

    def create_fresh(self, request: ResetRequest) -> RuntimeHandle:
        if type(request) is not ResetRequest:
            raise ContinuationError('CC_INVALID_PAYLOAD', 'ResetRequest required')
        bundle = self.registry.get(request.profile_id)
        cleanup = CleanupLedger()
        try:
            parts = bundle.fresh_factory.create_episode(request, cleanup)
            if type(parts) is not RuntimeParts:
                raise ContinuationError('CC_CONFORMANCE_FAILED', 'factory did not return RuntimeParts')
            parts.metadata.update({
                'policy_context': freeze_value(request.policy_context),
                'sampling_context': freeze_value(request.sampling_context),
            })
            parts.metadata.setdefault('config_sha256', digest(request.model_config))
            parts.refs.identity_snapshot()
            self._admit(bundle, parts)
            # Fully validate the initial state, but do not issue a capture claim.
            logical = {'family_id': request.run_id, 'prefix_id': 'reset',
                       'policy_context': request.policy_context,
                       'sampling_context': request.sampling_context}
            encode_snapshot(self._payload(bundle, parts, logical), registry=self.registry)
            return self._publish_runtime(bundle, parts, cleanup)
        except BaseException as exc:
            raise self._cleanup_failure(cleanup, exc, 'create-fresh') from exc

    def inspect(self, runtime: RuntimeHandle) -> CapabilityReport:
        try:
            with self._locked(runtime):
                self._require_ready(runtime)
                self.registry.resolve(runtime._bundle.profile)
                reasons = self._violations(runtime._bundle, runtime._parts)
                obligations = runtime._bundle.profile.obligations
                unknown = [item for item in obligations
                           if item.evidence_status not in ('tested', 'verified')]
                status = 'unsupported' if reasons else 'unverified' if unknown else 'supported'
                if not reasons and unknown:
                    reasons = tuple(Violation('CC_UNVERIFIED_OBLIGATION', obligation_id=item.obligation_id,
                                              message='declared obligation is not tested') for item in unknown)
                return CapabilityReport(
                    status,
                    reasons, runtime.profile_id,
                    obligation_status={item.obligation_id: item.evidence_status for item in obligations},
                )
        except ContinuationError as exc:
            if exc.code not in ('CC_BUSY', 'CC_INVALID_BOUNDARY', 'CC_UNSUPPORTED_PROFILE'):
                raise
            return CapabilityReport('busy' if exc.code == 'CC_BUSY' else 'unsupported',
                                    (Violation(exc.code, message=str(exc)),))

    def semantic_view(self, runtime: RuntimeHandle):
        with self._locked(runtime):
            self._require_ready(runtime)
            self._admit(runtime._bundle, runtime._parts)
            return freeze_value(self._states(runtime._bundle, runtime._parts))

    def capture(self, runtime: RuntimeHandle, request: CaptureRequest) -> ContinuationSnapshot:
        if type(request) is not CaptureRequest:
            raise ContinuationError('CC_INVALID_PAYLOAD', 'CaptureRequest required')
        with self._locked(runtime):
            self._require_ready(runtime)
            bundle, parts = runtime._bundle, runtime._parts
            if request.expected_profile_id != runtime.profile_id:
                raise ContinuationError('CC_INCOMPATIBLE_IDENTITY', 'capture profile differs')
            if request.expected_boundary_cursor != bundle.boundary.cursor(parts):
                raise ContinuationError('CC_STALE_BOUNDARY', 'capture cursor differs')
            if thaw_value(request.policy_context) != thaw_value(parts.metadata['policy_context']):
                raise ContinuationError('CC_INCOMPATIBLE_IDENTITY', 'capture policy differs')
            self._admit(bundle, parts)
            runtime._state = 'CAPTURING'
            before = None
            try:
                references = parts.refs.identity_snapshot()
                states = self._states(bundle, parts)
                before = canonical_bytes(states)
                logical = {'family_id': request.family_id, 'prefix_id': request.prefix_id,
                           'policy_context': request.policy_context,
                           'sampling_context': parts.metadata['sampling_context']}
                result = encode_snapshot(self._payload(bundle, parts, logical, states=states),
                                         registry=self.registry)
                return result
            finally:
                try:
                    parts.refs.assert_stable(references)
                    if before is None or canonical_bytes(self._states(bundle, parts)) != before:
                        raise ContinuationError('CC_CAPTURE_MUTATED', 'source changed during capture')
                    self._admit(bundle, parts)
                except BaseException as exc:
                    runtime._state = 'INVALID'
                    runtime._admission_witness = None
                    raise ContinuationError('CC_CAPTURE_MUTATED', 'source invariance could not be confirmed',
                                            phase='capture', state_disposition='INVALID') from exc
                else:
                    runtime._state = 'READY'

    def restore(self, snapshot: ContinuationSnapshot | bytes, branch: BranchContext) -> RuntimeHandle:
        if type(branch) is not BranchContext:
            raise ContinuationError('CC_INVALID_PAYLOAD', 'BranchContext required')
        # All wire/provider/composition checks precede any candidate allocation.
        with decode_checked(snapshot, registry=self.registry) as checked:
            return self._restore_checked(checked, branch)

    def _restore_checked(self, checked, branch):
        payload = checked.working_copy()
        # Never pass this detached expected value to extension callbacks. A
        # callback must not rewrite both the candidate and its comparison basis.
        expected = checked.working_copy()
        logical = payload['logical_context']
        if (branch.family_id != logical['family_id'] or branch.prefix_id != logical['prefix_id']
                or thaw_value(branch.policy_context) != logical['policy_context']):
            raise ContinuationError('CC_INCOMPATIBLE_IDENTITY', 'branch prefix/family/policy differs')
        bundle = checked.context.resolve('restore-branch')
        checked.context.advance_phase('branch-and-topology-callbacks')
        if bundle.boundary.validate_branch(logical, branch) is not None:
            raise ContinuationError('CC_INVALID_PAYLOAD', 'branch validator must return None or raise')
        if payload['topology'] != bundle.model.topology():
            raise ContinuationError('CC_INCOMPATIBLE_IDENTITY', 'topology differs from installed model declaration')
        checked.assert_unchanged(payload)
        cleanup = CleanupLedger()
        try:
            checked.context.advance_phase('candidate-restore-callbacks')
            engine_state, model_state, boundary_state = (
                payload['engine_state'], payload['model_state'], payload['boundary_state'])
            engine = bundle.engine.allocate_empty(engine_state['construction'], cleanup)
            services = bundle.engine.services(engine)
            graph = bundle.model.allocate_shell(model_state['construction'], services, cleanup)
            state_handle = bundle.boundary.allocate_state(boundary_state['descriptor'], cleanup)
            refs = ReferenceRegistry(payload['topology'], bundle.model.reference_objects(graph))
            bundle.engine.attach(engine, graph, refs)
            references = refs.identity_snapshot()
            bundle.model.restore_into(graph, model_state)
            bundle.model.rebind(graph, refs, services)
            refs.assert_stable(references)
            bundle.engine.restore_into(engine, engine_state, refs)
            binding = bundle.model.make_binding(graph, services, state_handle)
            env = bundle.boundary.bind_uninitialized(binding, engine, state_handle,
                                                      boundary_state['descriptor'], cleanup, branch=branch)
            bundle.boundary.restore_into(env, state_handle, boundary_state)
            parts = RuntimeParts(engine, graph, binding, env, state_handle, refs, {
                'config_sha256': payload['identities']['config_sha256'],
                'policy_context': freeze_value(branch.policy_context),
                'sampling_context': freeze_value(branch.sampling_context),
                'family_id': branch.family_id, 'prefix_id': branch.prefix_id,
                'branch_id': branch.branch_id,
            })
            bundle.engine.validate_restored(engine, engine_state, refs)
            bundle.model.validate_restored(graph, engine_state)
            bundle.boundary.validate_restored(env, model_state, engine_state,
                                             expected_state=boundary_state, branch=branch)
            refs.assert_stable(references)
            self._admit(bundle, parts)
            checked.assert_unchanged(payload)
            states = self._states(bundle, parts)
            if (states['engine_state'] != expected['engine_state']
                    or states['model_state'] != expected['model_state']):
                raise ContinuationError('CC_CONFORMANCE_FAILED', 'restored engine/model values differ')
            updated_logical = dict(logical, sampling_context=thaw_value(branch.sampling_context))
            encode_snapshot(self._payload(bundle, parts, updated_logical, states=states), registry=self.registry)
            checked.assert_unchanged(payload)
            return self._publish_runtime(bundle, parts, cleanup)
        except BaseException as exc:
            raise self._cleanup_failure(cleanup, exc, 'restore') from exc
