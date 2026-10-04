"""SDK invocation ledger and object-knowledge fences, separate from telemetry."""

import asyncio
import threading
from typing import Any


class OwnershipCallbacks:
    def __init__(self) -> None:
        self.enabled = False
        self.scope: tuple[int, int] | None = None
        self.dispatch_fence = threading.RLock()
        self.async_fence = asyncio.Lock()
        self.active_task: asyncio.Task[Any] | None = None
        self._lock = threading.RLock()
        self._latest: dict[int, int] = {}
        self._retired: dict[int, int] = {}
        self._invoked: set[tuple[int, int, int]] = set()
        self._pending: dict[tuple[int, int, int], Any] = {}

    @staticmethod
    def key(receipt: Any) -> tuple[int, int, int]:
        return (
            int(receipt.federation_generation),
            int(receipt.callback_epoch),
            int(receipt.callback_id),
        )

    def observe(self, event: Any) -> None:
        from .events import DiscoverObjectInstance

        if isinstance(event, DiscoverObjectInstance):
            epoch = int(getattr(event, "_object_knowledge_epoch", 0))
            with self._lock:
                self._latest[event.object_handle] = max(
                    epoch, self._latest.get(event.object_handle, 0)
                )

    def retire(self, object_handle: int, epoch: int) -> None:
        with self._lock:
            self._retired[object_handle] = max(epoch, self._retired.get(object_handle, 0))

    def begin(self, event: Any) -> bool:
        from .events import (
            AttributeOwnershipAcquisitionNotification,
            AttributeOwnershipUnavailable,
            ConfirmAttributeOwnershipAcquisitionCancellation,
            DiscoverObjectInstance,
            RequestAttributeOwnershipAssumption,
        )

        with self._lock:
            if isinstance(event, (DiscoverObjectInstance, RequestAttributeOwnershipAssumption)):
                epoch = int(getattr(event, "_object_knowledge_epoch", 0))
                obj = event.object_handle
                if epoch and (
                    epoch <= self._retired.get(obj, 0) or epoch < self._latest.get(obj, 0)
                ):
                    return False
            receipt = getattr(event, "_ownership_receipt", None)
            if receipt is None:
                return True
            if (
                not self.enabled
                or self.scope != (int(receipt.federation_generation), int(receipt.callback_epoch))
                or int(receipt.callback_id) <= 0
                or int(receipt.delivery_attempt) <= 0
                or not isinstance(
                    event,
                    (
                        AttributeOwnershipAcquisitionNotification,
                        AttributeOwnershipUnavailable,
                        ConfirmAttributeOwnershipAcquisitionCancellation,
                    ),
                )
            ):
                raise RuntimeError("stale, invalid, or unnegotiated ownership callback receipt")
            key = self.key(receipt)
            self._pending[key] = receipt
            if key in self._invoked:
                return False
            self._invoked.add(key)
            return True

    def pending(self) -> list[Any]:
        with self._lock:
            return list(self._pending.values())

    def entry_failed(self, event: Any) -> None:
        receipt = getattr(event, "_ownership_receipt", None)
        if receipt is not None:
            with self._lock:
                key = self.key(receipt)
                self._invoked.discard(key)
                self._pending.pop(key, None)

    def acknowledged(self, receipt: Any) -> None:
        with self._lock:
            key = self.key(receipt)
            if self._pending.get(key) is receipt:
                del self._pending[key]

    def reset_knowledge(self) -> None:
        with self._lock:
            self._latest.clear()
            self._retired.clear()

    def restore_scope(self, generation: int, epoch: int) -> None:
        with self._lock:
            self._latest.clear()
            self._retired.clear()
            self._pending.clear()
            self._invoked.clear()
            self.scope = (generation, epoch)
