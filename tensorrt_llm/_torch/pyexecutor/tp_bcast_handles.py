# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request broadcast that shares tensors with local tensor-parallel ranks.

Rank 0 exports each eligible tensor once per receiving rank through
``SharedTensorContainer`` (CPU shared memory or CUDA IPC) and broadcasts the
small handles instead of the tensor bytes. Peers map the handles to local views.

Lifetime: every export takes one reference. A peer that maps a handle releases
that reference when its view is freed, through the normal request teardown.
One acknowledgement allgather per broadcast that carries handles tells rank 0
which handles were not mapped, and rank 0 releases those references itself.

Failure handling: an export or mapping error does not fail the request. The
batch is sent again as bytes, and the helper stays on the byte path for the
rest of the run. A payload that cannot be pickled still raises, as it does
on the byte path.

Shared tensors are views of the same memory on every rank, so consumers must
not modify them in place.
"""

import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Tuple

import torch
from torch.utils._pytree import tree_map

from tensorrt_llm.logger import logger

from ..distributed.communicator import Distributed, SerializationError
from ..shared_tensor import SharedTensorContainer
from ..shared_tensor.shared_tensor import DTYPE_MAPPING

_SHAREABLE_DTYPES = frozenset(DTYPE_MAPPING.values())


@dataclass
class _TensorHandles:
    """Exported handles of one tensor, one per receiving rank (rank 1 first)."""

    handles: Tuple[Dict[str, Any], ...]


def _map_tensors(value: Any, operation: Callable[[Any], Any]) -> Any:
    return tree_map(
        operation, value, is_leaf=lambda item: isinstance(item, (_TensorHandles, torch.Size))
    )


def _can_share(tensor: torch.Tensor) -> bool:
    """Whether a tensor can be sent as a handle; other tensors are pickled as bytes."""
    return (
        tensor.layout == torch.strided
        and tensor.device.type in ("cpu", "cuda")
        and tensor.dtype in _SHAREABLE_DTYPES
        and tensor.numel() > 0
        and not tensor.requires_grad
        and not tensor.is_neg()
    )


class TpBroadcastHandles:
    """Broadcast requests to local TP ranks with shared tensor storage.

    It is a drop-in replacement for ``dist.broadcast((requests, py_objects),
    root=0)`` when all ranks are tensor-parallel peers of one host. Only the
    tensors inside the named request attributes are shared. All other values,
    and tensors that cannot be shared, are pickled as before.

    Collectives: one hostname allgather on first use. Then, per call, the
    payload broadcast, plus one acknowledgement allgather when the payload
    carries handles. Iterations without shareable tensors add no collective.
    """

    def __init__(self, dist: Distributed, tensor_attributes: Iterable[str]) -> None:
        """Create the helper.

        Args:
            dist: Transport whose ranks are all TP peers.
            tensor_attributes: Names of the request attributes in
                ``py_objects`` whose tensors are shared.
        """
        self.dist = dist
        self.tensor_attributes: FrozenSet[str] = frozenset(tensor_attributes)
        # None until the first call checks that all ranks share one host.
        self.enabled: Optional[bool] = None

    def _disable(self, reason: str) -> None:
        self.enabled = False
        if self.dist.rank == 0:
            logger.warning(f"TP shared-tensor broadcast disabled, using byte transport: {reason}")

    def _transform(
        self, py_objects: Tuple[Tuple[str, Any], ...], operation: Callable[[Any], Any]
    ) -> Tuple[Tuple[str, Any], ...]:
        return tuple(
            (name, _map_tensors(value, operation) if name in self.tensor_attributes else value)
            for name, value in py_objects
        )

    def broadcast(
        self, requests: List, py_objects: Optional[Tuple[Tuple[str, Any], ...]]
    ) -> Tuple[List, Optional[Tuple]]:
        """Broadcast requests and their Python objects from rank 0.

        This is a collective call. Every rank must call it.

        Args:
            requests: Requests on rank 0. Ignored on other ranks.
            py_objects: ``(attribute_name, {request_id: value})`` pairs on
                rank 0. Ignored on other ranks.

        Returns:
            Rank 0 gets its own ``(requests, py_objects)`` back. Other ranks
            get the received requests and Python objects, with shared tensors
            mapped to local views.

        Raises:
            SerializationError: On rank 0, if the payload cannot be pickled.
            RuntimeError: On other ranks, in the same case.
        """
        dist = self.dist
        if self.enabled is None:
            self.enabled = len(set(dist.allgather(socket.gethostname()))) == 1
            if dist.rank == 0:
                logger.info(
                    f"TP shared-tensor broadcast: enabled={self.enabled}, ranks={dist.world_size}"
                )
        if not self.enabled:
            return dist.broadcast((requests, py_objects), root=0)

        peers = dist.world_size - 1
        pending: List[SharedTensorContainer] = []
        # Rank 0 keeps local copies alive until the peers have mapped them.
        owners: List[torch.Tensor] = []
        wire, export_error = py_objects, None

        def export(value: Any) -> Any:
            if not isinstance(value, torch.Tensor) or not _can_share(value):
                return value
            owner, handles = value, []
            for _ in range(peers):
                try:
                    container = SharedTensorContainer.from_tensor(owner)
                except RuntimeError:
                    if owner is not value or not value.is_cuda:
                        raise
                    # A CUDA allocation imported from another process cannot
                    # be exported again, so export a local copy instead.
                    owner = value.clone()
                    owners.append(owner)
                    container = SharedTensorContainer.from_tensor(owner)
                # CUDA takes its export reference in from_tensor, CPU in dump_to_dict.
                if owner.is_cuda:
                    pending.append(container)
                handle = container.dump_to_dict()
                if not owner.is_cuda:
                    pending.append(container)
                handles.append(handle)
            return _TensorHandles(tuple(handles))

        if dist.rank == 0:
            try:
                wire = self._transform(py_objects, export)
            except Exception as exc:
                for container in pending:
                    container.release_unconsumed()
                pending.clear()
                wire, export_error = py_objects, repr(exc)

        try:
            result, wire, has_handles, export_error = dist.broadcast(
                (requests, wire, bool(pending), export_error), root=0
            )
        except SerializationError:
            # Rank 0 fails before any handle leaves the rank.
            for container in pending:
                container.release_unconsumed()
            raise
        if export_error:
            self._disable("export failed: " + export_error)
        if not has_handles:
            return (requests, py_objects) if dist.rank == 0 else (result, wire)

        consumed: List[bool] = []
        errors: List[str] = []

        def restore(value: Any) -> Any:
            if not isinstance(value, _TensorHandles):
                return value
            consumed.append(False)
            try:
                container = SharedTensorContainer.from_dict(value.handles[dist.rank - 1])
                consumed[-1] = container.consumes_reference_on_from_dict
                tensor = container.get_local_view()
                consumed[-1] = True
                return tensor
            except Exception as exc:
                # Visit every handle, even after an earlier mapping failed.
                errors.append(repr(exc))
                return None

        if dist.rank != 0:
            wire = self._transform(wire, restore)
        acknowledgements = dist.allgather((consumed, errors))
        if dist.rank == 0:
            for index, container in enumerate(pending):
                peer, slot = index % peers + 1, index // peers
                if not acknowledgements[peer][0][slot]:
                    container.release_unconsumed()
        failures = [message for _, messages in acknowledgements for message in messages]
        if failures:
            # Every rank sees the same failures, so all ranks take this branch.
            self._disable("mapping failed: " + repr(failures))
            wire = dist.broadcast(py_objects if dist.rank == 0 else None, root=0)
        # Mapped storage now belongs to each request. Completion, cancellation
        # and error teardown drop it after the final use.
        return (requests, py_objects) if dist.rank == 0 else (result, wire)
