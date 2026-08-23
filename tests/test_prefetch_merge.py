"""MoEExpertReader prefetch semantics (Bug 1) and backfill safety (Bug 3),
run against a real tiny streaming-format model on disk."""

import numpy as np
import pytest

from mlx_expert_sniper.expert_io import MoEExpertReader

from conftest import NUM_LAYERS, expert_bytes


@pytest.fixture
def reader(model_dir):
    r = MoEExpertReader(model_dir, num_layers=NUM_LAYERS, num_workers=4,
                        cache_size=8)
    yield r
    r.close()


def test_second_prefetch_call_merges_not_overwrites(reader):
    reader.prefetch_experts(1, [0, 1])      # "predicted" experts
    reader.prefetch_experts(1, [2, 3])      # "active" experts — must merge
    assert sorted(reader.prefetch_futures[1]) == [0, 1, 2, 3]


def test_prefetch_dedupes_in_flight_and_cached(reader):
    reader.prefetch_experts(1, [0, 1])
    first = dict(reader.prefetch_futures[1])
    reader.prefetch_experts(1, [1, 2])      # 1 already in flight
    assert reader.prefetch_futures[1][1] is first[1]
    reader.get_experts(1, [0])              # 0 now cached
    reader.prefetch_experts(1, [0, 3])      # 0 cached → skipped
    assert 0 not in reader.prefetch_futures[1]
    assert 3 in reader.prefetch_futures[1]


def test_prefetched_experts_served_and_counted(reader):
    reader.prefetch_experts(0, [4, 5])
    experts = reader.get_experts(0, [4, 5])
    assert set(experts) == {4, 5}
    assert reader.prefetch_hits == 2
    # correct bytes parsed for the right expert
    import mlx.core as mx
    got = np.array(experts[4]["switch_mlp.gate_proj.weight"].astype(mx.float32))
    want = np.frombuffer(expert_bytes(0, 4), dtype=np.float16)[:16] \
        .astype(np.float32).reshape(4, 4)
    assert np.array_equal(got, want)


def test_leftover_prefetch_parked_in_victim_buffer(reader):
    reader.prefetch_experts(2, [6, 7, 8])
    for fut in reader.prefetch_futures[2].values():
        fut.result()                        # ensure reads complete
    reader.get_experts(2, [6])              # 7, 8 are leftovers
    # Speculation must NOT pollute the main LRU...
    assert not reader.lru.contains(2, 7)
    assert (2, 7) in reader.victim and (2, 8) in reader.victim
    assert reader.prefetch_futures.get(2) is None  # popped
    # ...but a correct speculation is promoted on use, without an SSD read
    ssd_before = reader.reads - reader.cache_hits
    reader.get_experts(2, [7])
    assert reader.victim_hits == 1
    assert reader.lru.contains(2, 7)
    assert (2, 7) not in reader.victim


def test_reset_prefetch_clears_all(reader):
    reader.prefetch_experts(0, [1])
    reader.prefetch_experts(1, [2])
    reader.reset_prefetch()
    assert reader.prefetch_futures == {}


def test_cache_hit_rate_reflects_serving_only(reader):
    reader.get_experts(0, [0, 1])           # misses
    reader.get_experts(0, [0, 1])           # hits
    assert reader.lru.hits == 2 and reader.lru.misses == 2
    # probing must not move the needle (the old 256-wide probe bug)
    for eid in range(16):
        reader.lru.contains(0, eid)
    assert reader.lru.hits == 2 and reader.lru.misses == 2


def test_backfill_runs_on_separate_executor_and_counts_errors(reader):
    from concurrent.futures import Future
    bad = Future()
    bad.set_exception(IOError("simulated read failure"))
    reader._schedule_backfill(0, {9: bad})
    reader.backfill_executor.shutdown(wait=True)
    reader.backfill_executor = type(reader.backfill_executor)(max_workers=1)
    assert reader.backfill_errors == 1
    assert not reader.lru.contains(0, 9)


def test_backfill_success_populates_cache(reader):
    fut = reader.executor.submit(reader._read_expert, 0, 10)
    reader._schedule_backfill(0, {10: fut})
    reader.backfill_executor.shutdown(wait=True)
    reader.backfill_executor = type(reader.backfill_executor)(max_workers=1)
    assert reader.lru.contains(0, 10)
    assert reader.backfill_errors == 0
