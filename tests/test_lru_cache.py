"""LRUExpertCache: probes must not mutate stats or recency; eviction must be
true LRU; the per-layer index must track put/evict; concurrent access must not
corrupt the structures."""

import threading

from mlx_expert_sniper.expert_io import LRUExpertCache


def test_get_counts_hits_and_misses():
    c = LRUExpertCache(max_experts=4)
    c.put(0, 1, "a")
    assert c.get(0, 1) == "a"
    assert c.get(0, 2) is None
    assert c.hits == 1 and c.misses == 1


def test_contains_and_peek_do_not_mutate():
    c = LRUExpertCache(max_experts=4)
    c.put(0, 1, "a")
    for _ in range(50):
        assert c.contains(0, 1)
        assert not c.contains(0, 9)
        assert c.peek(0, 1) == "a"
        assert c.peek(0, 9) is None
    assert c.hits == 0 and c.misses == 0


def test_probes_do_not_disturb_lru_order():
    c = LRUExpertCache(max_experts=2)
    c.put(0, 1, "a")
    c.put(0, 2, "b")
    c.get(0, 1)          # 1 is now most-recent
    # Probing 2 must NOT refresh it (the old bug: probing sorted eviction by id)
    for _ in range(10):
        c.contains(0, 2)
        c.peek(0, 2)
    c.put(0, 3, "c")     # evicts the true LRU: 2, not 1
    assert c.contains(0, 1)
    assert not c.contains(0, 2)
    assert c.contains(0, 3)


def test_eviction_is_lru_via_get():
    c = LRUExpertCache(max_experts=2)
    c.put(0, 1, "a")
    c.put(0, 2, "b")
    c.get(0, 1)          # refresh 1
    c.put(0, 3, "c")
    assert c.contains(0, 1) and not c.contains(0, 2)


def test_cached_ids_tracks_put_and_evict():
    c = LRUExpertCache(max_experts=2)
    c.put(5, 10, "a")
    c.put(5, 11, "b")
    assert sorted(c.cached_ids(5)) == [10, 11]
    assert c.cached_ids(6) == []
    c.put(6, 20, "c")    # evicts (5, 10)
    assert sorted(c.cached_ids(5)) == [11]
    assert c.cached_ids(6) == [20]


def test_put_existing_key_updates_value_without_evicting():
    c = LRUExpertCache(max_experts=2)
    c.put(0, 1, "a")
    c.put(0, 2, "b")
    c.put(0, 1, "a2")    # update, no eviction
    assert c.peek(0, 1) == "a2" and c.contains(0, 2)


def test_concurrent_put_get_probe_does_not_corrupt():
    c = LRUExpertCache(max_experts=32)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            c.put(i % 4, i % 64, i)
            i += 1

    def reader():
        i = 0
        while not stop.is_set():
            try:
                c.get(i % 4, i % 64)
                c.contains(i % 4, (i + 1) % 64)
                c.cached_ids(i % 4)
            except Exception as e:  # pragma: no cover
                errors.append(e)
                stop.set()
            i += 1

    threads = [threading.Thread(target=writer) for _ in range(2)] + \
              [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()
    stop.wait(timeout=1.0)
    stop.set()
    for t in threads:
        t.join()
    assert not errors
    assert len(c.cache) <= 32
    # per-layer index must exactly mirror the cache
    indexed = {(l, e) for l, ids in c._by_layer.items() for e in ids}
    assert indexed == set(c.cache.keys())
