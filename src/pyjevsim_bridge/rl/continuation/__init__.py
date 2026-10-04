"""Opt-in continuation contracts; no implicit provider registration."""
from .contracts import (
    BranchContext, CapabilityReport, CaptureRequest, CleanupReceipt,
    ContinuationBundle, ContinuationError, ContinuationSnapshot,
    ParserLimits, ProfileDescriptor, ProviderIdentity, ResetRequest,
    StateObligation,
)
from .coordinator import ContinuationCoordinator, RuntimeHandle
from .registry import ContinuationRegistry

__all__ = [
    'BranchContext', 'CapabilityReport', 'CaptureRequest', 'CleanupReceipt',
    'ContinuationBundle', 'ContinuationCoordinator', 'ContinuationError',
    'ContinuationRegistry', 'ContinuationSnapshot', 'ParserLimits',
    'ProfileDescriptor', 'ProviderIdentity', 'ResetRequest', 'RuntimeHandle',
    'StateObligation',
]
