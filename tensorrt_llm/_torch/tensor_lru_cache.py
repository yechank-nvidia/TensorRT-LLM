# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Generic, NamedTuple, TypeVar

import torch

from tensorrt_llm.logger import logger

K = TypeVar("K", bound=Hashable)


class CacheEntryState(Enum):
    """Lifecycle state of a cache key."""

    # ABSENT is used by strict operations to require a missing key. Stored
    # entries are either RESERVED or READY.
    ABSENT = auto()
    # Expected bytes and references exist, but no tensor has been committed.
    RESERVED = auto()
    # The cache owns a tensor that callers may read.
    READY = auto()


class CacheAcquireResult(Enum):
    """How `acquire` satisfied a successful request."""

    # A READY entry was found and referenced.
    READY_HIT = auto()
    # A missing key became a new RESERVED entry.
    NEW_RESERVATION = auto()
    # An existing RESERVED entry gained another reference.
    RESERVATION_HIT = auto()


@dataclass
class _Entry:
    state: CacheEntryState
    size_bytes: int
    reference_count: int
    retain_after_release: bool
    value: torch.Tensor | None = None
    # CUDA event recorded on the producing stream right after the clone in `put`. Consumers on a
    # different stream wait on it before reading `value`. `None` for CPU tensors or when the cache
    # is not stream-aware.
    producer_event: torch.cuda.Event | None = None


class TensorLRUCacheStats(NamedTuple):
    max_bytes: int
    current_bytes: int
    reserved_bytes: int
    pinned_bytes: int
    item_count: int
    hits: int
    misses: int
    insertions: int
    replacements: int
    evictions: int
    rejected_insertions: int
    producer_misses: int
    inflight_deduplications: int
    hit_rate: float


@dataclass
class _CacheCounters:
    hits: int = 0
    misses: int = 0
    insertions: int = 0
    replacements: int = 0
    evictions: int = 0
    rejected_insertions: int = 0
    producer_misses: int = 0
    inflight_deduplications: int = 0

    @property
    def hit_rate(self) -> float:
        total_gets = self.hits + self.misses
        return self.hits / total_gets if total_gets else 0.0


class TensorLRUCache(Generic[K]):
    """Thread-safe LRU cache from hashable keys to tensor values.

    Size accounting uses logical tensor bytes: `tensor.numel() * tensor.element_size()`.
    Returned tensors alias the cache-owned tensor objects. Callers must treat them as immutable
    while they remain cache-owned.

    The cache owns a detached copy rather than the caller's tensor or view. This prevents later
    caller mutations from changing a cached value and prevents a small cached view from retaining
    the caller's larger backing allocation. The copy is made before acquiring the lock and before
    replacement or eviction, preserving existing cache entries if copying fails. Consequently,
    `max_bytes` bounds steady-state logical cache contents, not peak allocation: insertion
    temporarily needs both the source tensor and its copy and may exceed the cache limit until
    eviction completes.

    Managed entries add a RESERVED state and reference-counted READY state to the original cache:

    1. `acquire` adds a reference and creates a `RESERVED` entry on a miss.
    2. `ensure_capacity` evicts only unreferenced `READY` entries before selected
       producers materialize their outputs.
    3. Strict `put` commits a `RESERVED` entry as `READY` without choosing more
       eviction victims.
    4. `get` reads a `READY` tensor without changing its reference count.
    5. `release` drops the reference and applies the entry's retention policy.

    Reservations and referenced READY entries cannot be evicted. `pinned_bytes` is the logical
    size of referenced READY tensors, not CUDA pinned host memory.

    `pop` is intentionally separate from `release`: it physically removes an
    unreferenced entry without applying reference or retention policy. This is
    used when a caller must replay an exact removal or roll back a reservation.

    In CUDA-stream-aware mode, each entry owns the event recorded after its clone. Replacement,
    eviction, and clear drop that event with the entry; events are not reused because an evicted
    tensor may still have outstanding consumers on another stream.

    Args:
        max_bytes: Maximum logical tensor bytes held by this cache.
        name: Short label used in debug log messages.
        cuda_stream_aware: When enabled, synchronize CUDA tensor producers and consumers across
            streams and extend allocation lifetime through every consuming stream. CPU tensors are
            unaffected.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        name: str = "tensor_lru_cache",
        cuda_stream_aware: bool = False,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._max_bytes = max_bytes
        self._name = name
        self._cuda_stream_aware = cuda_stream_aware
        self._current_bytes = 0
        self._reserved_bytes = 0
        # Ready tensor bytes protected from eviction by live references, not CUDA pinned memory.
        self._pinned_bytes = 0
        self._items: OrderedDict[K, _Entry] = OrderedDict()
        self._lock = RLock()
        self._counters = _CacheCounters()

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def acquire(
        self,
        key: K,
        expected_bytes: int,
        *,
        retain_after_release: bool = True,
    ) -> CacheAcquireResult | None:
        """Acquire a reference, reserving a missing entry when needed.

        A READY hit and an existing reservation both gain one reference. A
        missing key becomes RESERVED and accounts `expected_bytes` until the
        producer commits its tensor with strict `put`.

        Args:
            key: Identity shared by producers and consumers of one tensor.
            expected_bytes: Exact tensor bytes that a missing key will produce.
            retain_after_release: Whether a READY entry remains reusable after
                its final reference is released.

        Returns:
            How the reference was acquired, or `None` when live references and
            reservations temporarily consume all capacity.

        Raises:
            ValueError: If the requested size is invalid or existing metadata
                conflicts with the request.
        """
        if expected_bytes <= 0:
            raise ValueError("expected_bytes must be positive")
        if expected_bytes > self._max_bytes:
            raise ValueError(
                f"expected_bytes ({expected_bytes}) exceeds cache capacity ({self._max_bytes})"
            )

        with self._lock:
            entry = self._items.get(key)
            if entry is not None:
                if entry.size_bytes != expected_bytes:
                    raise ValueError(
                        f"existing cache entry size ({entry.size_bytes}) does not match "
                        f"expected_bytes ({expected_bytes})"
                    )
                if entry.retain_after_release != retain_after_release:
                    raise ValueError("existing cache entry retention policy does not match")

                if entry.state is CacheEntryState.RESERVED:
                    entry.reference_count += 1
                    self._counters.inflight_deduplications += 1
                    return CacheAcquireResult.RESERVATION_HIT
                if entry.state is not CacheEntryState.READY:
                    raise RuntimeError(f"unexpected cache entry state: {entry.state}")

                if entry.reference_count == 0:
                    if self._in_use_bytes + entry.size_bytes > self._max_bytes:
                        return None
                    self._pinned_bytes += entry.size_bytes
                entry.reference_count += 1
                self._items.move_to_end(key)
                self._counters.hits += 1
                return CacheAcquireResult.READY_HIT

            if self._in_use_bytes + expected_bytes > self._max_bytes:
                return None

            self._items[key] = _Entry(
                state=CacheEntryState.RESERVED,
                size_bytes=expected_bytes,
                reference_count=1,
                retain_after_release=retain_after_release,
            )
            self._reserved_bytes += expected_bytes
            self._counters.misses += 1
            self._counters.producer_misses += 1
            return CacheAcquireResult.NEW_RESERVATION

    def get(self, key: K, *, record_stats: bool = True) -> torch.Tensor | None:
        """Return a cache-owned, immutable tensor and promote it to most-recently-used.

        The returned tensor aliases the cached value. Callers must not mutate it.
        Only READY entries are returned. `get` does not acquire a managed
        reference; `record_stats=False` suppresses its hit/miss accounting.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None or entry.state is not CacheEntryState.READY:
                if record_stats:
                    self._counters.misses += 1
                return None

            if record_stats:
                self._counters.hits += 1
            self._items.move_to_end(key)
            self._prepare_for_current_stream(entry)
            assert entry.value is not None
            return entry.value

    def put(
        self,
        key: K,
        value: torch.Tensor,
        *,
        expected_state: CacheEntryState | None = None,
        expected_bytes: int | None = None,
    ) -> bool:
        """Insert or replace a tensor.

        Returns `False` and leaves the cache unchanged when `value` is larger than the full
        cache capacity.

        Managed callers set `expected_state=RESERVED` to commit an acquired
        producer output, or `ABSENT` to replay an exact remote insertion. The
        expected size must match and `ensure_capacity` must already have made
        room; these strict puts never choose eviction victims.
        """
        size_bytes = self._tensor_size_bytes(value)

        if expected_state not in (
            None,
            CacheEntryState.ABSENT,
            CacheEntryState.RESERVED,
        ):
            raise ValueError("strict put only supports ABSENT or RESERVED expected state")
        if expected_state is None and expected_bytes is not None:
            raise ValueError("expected_bytes requires expected_state")
        if expected_state is CacheEntryState.ABSENT and expected_bytes is None:
            raise ValueError("strict ABSENT insertion requires expected_bytes")
        if expected_bytes is not None and size_bytes != expected_bytes:
            raise ValueError(
                f"tensor size ({size_bytes}) does not match expected_bytes ({expected_bytes})"
            )
        if size_bytes > self._max_bytes:
            if expected_state is not None:
                raise ValueError(
                    f"tensor size ({size_bytes}) exceeds cache capacity ({self._max_bytes})"
                )
            with self._lock:
                self._counters.rejected_insertions += 1
            logger.debug(
                f"{self._name}: rejected oversized tensor insertion, size_bytes={size_bytes}, "
                f"max_bytes={self._max_bytes}"
            )
            return False

        stored_value = None
        producer_event = None
        if expected_state is None:
            stored_value, producer_event = self._clone_for_storage(value)

        with self._lock:
            if expected_state is not None:
                return self._put_with_expected_state(key, value, size_bytes, expected_state)

            old_entry = self._items.get(key)
            if old_entry is not None:
                if old_entry.state is not CacheEntryState.READY:
                    raise RuntimeError("cannot replace a reserved cache entry")
                if old_entry.reference_count:
                    raise RuntimeError("cannot replace a referenced cache entry")
                del self._items[key]
                self._current_bytes -= old_entry.size_bytes
                self._counters.replacements += 1
            else:
                self._counters.insertions += 1

            assert stored_value is not None
            self._items[key] = _Entry(
                state=CacheEntryState.READY,
                size_bytes=size_bytes,
                reference_count=0,
                retain_after_release=True,
                value=stored_value,
                producer_event=producer_event,
            )
            self._current_bytes += size_bytes

            evicted_count, evicted_bytes = self._evict_until_within_limit()
            if evicted_count:
                self._counters.evictions += evicted_count
                logger.debug(
                    f"{self._name}: evicted {evicted_count} LRU entries, "
                    f"freed_bytes={evicted_bytes}, current_bytes={self._current_bytes}, "
                    f"max_bytes={self._max_bytes}"
                )
            return True

    def ensure_capacity(self, incoming_bytes: int) -> list[K]:
        """Ensure physical space for selected outputs before they are produced.

        `incoming_bytes` is the total size of outputs selected for the next
        strict puts. Only unreferenced READY entries are eligible LRU victims;
        reservations and referenced tensors are never removed. The returned
        keys let an authority replay the exact removals on peer caches. A
        failure leaves the cache unchanged.
        """
        if incoming_bytes < 0:
            raise ValueError("incoming_bytes must be non-negative")
        with self._lock:
            required_bytes = max(
                0,
                self._current_bytes + incoming_bytes - self._max_bytes,
            )
            if required_bytes == 0:
                return []

            victims: list[tuple[K, _Entry]] = []
            freed_bytes = 0
            for victim_key, victim in list(self._items.items()):
                if victim.state is not CacheEntryState.READY or victim.reference_count != 0:
                    continue
                victims.append((victim_key, victim))
                freed_bytes += victim.size_bytes
                if freed_bytes >= required_bytes:
                    break

            if freed_bytes < required_bytes:
                raise RuntimeError("cache does not have enough removable space")

            for victim_key, victim in victims:
                del self._items[victim_key]
                self._current_bytes -= victim.size_bytes
            self._counters.evictions += len(victims)
            return [victim_key for victim_key, _ in victims]

    def release(self, key: K) -> K | None:
        """Release one acquired reference and apply retention policy.

        A RESERVED entry disappears when its final producer or follower
        reference is released. A READY entry becomes evictable at zero
        references and remains cached when `retain_after_release` is true;
        otherwise it is removed immediately. Returns the key only when a READY
        tensor was physically removed, allowing an authority to replay that
        removal on peer caches.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None or entry.reference_count == 0:
                raise RuntimeError("cannot release an unreferenced cache entry")

            entry.reference_count -= 1
            if entry.reference_count:
                return None

            if entry.state is CacheEntryState.RESERVED:
                self._reserved_bytes -= entry.size_bytes
                del self._items[key]
                return None

            if entry.state is not CacheEntryState.READY:
                raise RuntimeError(f"unexpected cache entry state: {entry.state}")

            self._pinned_bytes -= entry.size_bytes
            if entry.retain_after_release:
                return None

            self._current_bytes -= entry.size_bytes
            del self._items[key]
            return key

    def pop(self, key: K, *, expected_state: CacheEntryState | None = None) -> torch.Tensor | None:
        """Remove one unreferenced key without applying release policy.

        `expected_state` makes a missing key or state mismatch an error for
        exact rollback and peer replay. Returns the READY tensor, or `None` for
        a removed reservation or an optional miss.
        """
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                if expected_state is not None:
                    raise RuntimeError("expected cache removal target is absent")
                return None
            if expected_state is not None and entry.state is not expected_state:
                raise RuntimeError(f"cache removal expected {expected_state}, found {entry.state}")
            if entry.reference_count:
                raise RuntimeError("cannot remove a referenced cache entry")

            del self._items[key]
            if entry.state is CacheEntryState.RESERVED:
                self._reserved_bytes -= entry.size_bytes
                return None

            self._current_bytes -= entry.size_bytes
            self._prepare_for_current_stream(entry)
            assert entry.value is not None
            return entry.value

    def clear(self) -> None:
        """Remove all entries unless a managed reference or reservation is active."""
        with self._lock:
            if self._in_use_bytes:
                raise RuntimeError("cannot clear cache with live references or reservations")
            self._items.clear()
            self._current_bytes = 0

    def stats(self) -> TensorLRUCacheStats:
        with self._lock:
            return TensorLRUCacheStats(
                max_bytes=self._max_bytes,
                current_bytes=self._current_bytes,
                reserved_bytes=self._reserved_bytes,
                pinned_bytes=self._pinned_bytes,
                item_count=len(self._items),
                hits=self._counters.hits,
                misses=self._counters.misses,
                insertions=self._counters.insertions,
                replacements=self._counters.replacements,
                evictions=self._counters.evictions,
                rejected_insertions=self._counters.rejected_insertions,
                producer_misses=self._counters.producer_misses,
                inflight_deduplications=self._counters.inflight_deduplications,
                hit_rate=self._counters.hit_rate,
            )

    def log_stats(self, reason: str) -> None:
        stats = self.stats()
        logger.debug(
            f"{self._name}: stats after {reason}: items={stats.item_count}, "
            f"bytes={stats.current_bytes}/{stats.max_bytes}, hits={stats.hits}, "
            f"misses={stats.misses}, hit_rate={stats.hit_rate:.3f}, "
            f"insertions={stats.insertions}, replacements={stats.replacements}, "
            f"evictions={stats.evictions}, rejected_insertions={stats.rejected_insertions}, "
            f"producer_misses={stats.producer_misses}, "
            f"inflight_deduplications={stats.inflight_deduplications}"
        )

    @property
    def _in_use_bytes(self) -> int:
        return self._reserved_bytes + self._pinned_bytes

    @staticmethod
    def _tensor_size_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _clone_for_storage(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Clone a value and record its producer event when stream-aware."""
        stored_value = value.detach().clone()
        producer_event = None
        if self._cuda_stream_aware and stored_value.is_cuda:
            producer_event = torch.cuda.Event()
            producer_event.record(torch.cuda.current_stream(stored_value.device))
        return stored_value, producer_event

    def _put_with_expected_state(
        self,
        key: K,
        value: torch.Tensor,
        size_bytes: int,
        expected_state: CacheEntryState,
    ) -> bool:
        """Apply a strict managed insertion after validating the expected state."""
        entry = self._items.get(key)
        actual_state = CacheEntryState.ABSENT if entry is None else entry.state
        if actual_state is not expected_state:
            raise RuntimeError(f"cache insertion expected {expected_state}, found {actual_state}")

        if expected_state is CacheEntryState.RESERVED:
            assert entry is not None
            if size_bytes != entry.size_bytes:
                raise ValueError(
                    f"tensor size ({size_bytes}) does not match reserved bytes ({entry.size_bytes})"
                )
            if self._current_bytes + size_bytes > self._max_bytes:
                raise RuntimeError("reserved entry does not have enough cache space")

            stored_value, producer_event = self._clone_for_storage(value)
            entry.state = CacheEntryState.READY
            entry.value = stored_value
            entry.producer_event = producer_event
            self._reserved_bytes -= size_bytes
            self._pinned_bytes += size_bytes
            self._current_bytes += size_bytes
            self._items.move_to_end(key)
            self._counters.insertions += 1
            return True

        assert expected_state is CacheEntryState.ABSENT
        if self._current_bytes + size_bytes > self._max_bytes:
            raise RuntimeError("remote cache insertion exceeds cache capacity")

        stored_value, producer_event = self._clone_for_storage(value)
        self._items[key] = _Entry(
            state=CacheEntryState.READY,
            size_bytes=size_bytes,
            reference_count=0,
            retain_after_release=True,
            value=stored_value,
            producer_event=producer_event,
        )
        self._current_bytes += size_bytes
        self._counters.insertions += 1
        return True

    def _prepare_for_current_stream(self, entry: _Entry) -> None:
        """Order and anchor a cached tensor for consumption on the current stream.

        Called from `get` / `pop` before returning an entry it:
        * makes the current (consuming) stream wait on the entry's producer event, so a cross-stream
          read observes fully-written data
        * calls `record_stream` on the consuming stream so the caching allocator will not reuse
          the storage while consumer-stream work is still pending, even if a later replacement or
          eviction drops the cache's own reference.
        """
        if entry.value is None:
            raise RuntimeError("cannot access a cache entry before its value is stored")
        if not self._cuda_stream_aware or not entry.value.is_cuda:
            return

        consumer_stream = torch.cuda.current_stream(entry.value.device)
        # The producer event orders the data dependency; `record_stream` separately guards lifetime.
        if entry.producer_event is not None:
            consumer_stream.wait_event(entry.producer_event)
        entry.value.record_stream(consumer_stream)

    def _evict_until_within_limit(self) -> tuple[int, int]:
        evicted_count = 0
        evicted_bytes = 0
        for key, entry in list(self._items.items()):
            if self._current_bytes <= self._max_bytes:
                break
            if entry.state is not CacheEntryState.READY or entry.reference_count:
                continue
            del self._items[key]
            self._current_bytes -= entry.size_bytes
            evicted_count += 1
            evicted_bytes += entry.size_bytes
        if self._current_bytes > self._max_bytes:
            raise RuntimeError("cache does not have enough removable space")
        return evicted_count, evicted_bytes
