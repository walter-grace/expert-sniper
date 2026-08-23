"""Rendezvous (HRW) placement over an anchored node roster.

Every participant computes expert ownership from the same published roster
— a plain ordered list of node ids — so no membership consensus is needed:
the roster IS the agreement. Adding or removing a node moves only ~1/N of
the experts (the defining property of rendezvous hashing).
"""
import hashlib


def owner(roster, expert_id):
    """The node id that owns this expert under the given roster."""
    def score(node_id):
        return hashlib.sha256(f"{node_id}:{expert_id}".encode()).digest()
    return max(roster, key=score)


def partition(roster, me, num_experts):
    """Sorted list of expert ids `me` owns under the roster."""
    if me not in roster:
        raise ValueError(f"{me!r} is not in the roster {roster}")
    return sorted(e for e in range(num_experts) if owner(roster, e) == me)


def plan(roster, num_experts):
    """Full assignment: {node_id: [expert ids]}."""
    out = {n: [] for n in roster}
    for e in range(num_experts):
        out[owner(roster, e)].append(e)
    return out
