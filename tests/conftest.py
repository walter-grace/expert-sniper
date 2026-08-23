"""Shared fixture: a tiny synthetic model in sniper streaming format.

Layer files are real (16 KB JSON header + fixed-size expert blocks) so
MoEExpertReader runs its actual pread path, just over kilobytes not gigabytes.
"""

import json
import struct

import numpy as np
import pytest

PAGE_SIZE = 16384
NUM_LAYERS = 3
NUM_EXPERTS = 16
GATE_SHAPE = [4, 4]   # 16 float16 values = 32 bytes
DOWN_SHAPE = [4, 4]
GATE_NBYTES = 32
DOWN_NBYTES = 32
EXPERT_BLOCK_SIZE = 64


def expert_bytes(layer_idx, eid):
    """Deterministic content so tests can verify which expert was parsed."""
    vals = np.arange(32, dtype=np.float16) + layer_idx * 1000 + eid
    return vals.tobytes()  # 64 bytes = gate (32) + down (32)


@pytest.fixture
def model_dir(tmp_path):
    header = {
        "layout": {
            "expert_block_size": EXPERT_BLOCK_SIZE,
            "data_start": PAGE_SIZE,
            "tensors": {
                "switch_mlp.gate_proj.weight": {
                    "inner_offset": 0, "nbytes": GATE_NBYTES,
                    "shape_per_expert": GATE_SHAPE, "dtype": "float16",
                },
                "switch_mlp.down_proj.weight": {
                    "inner_offset": GATE_NBYTES, "nbytes": DOWN_NBYTES,
                    "shape_per_expert": DOWN_SHAPE, "dtype": "float16",
                },
            },
        }
    }
    raw_header = json.dumps(header).encode()
    assert len(raw_header) < PAGE_SIZE
    for i in range(NUM_LAYERS):
        with open(tmp_path / f"layer_{i:02d}.bin", "wb") as f:
            f.write(raw_header.ljust(PAGE_SIZE, b"\x00"))
            for eid in range(NUM_EXPERTS):
                f.write(expert_bytes(i, eid))
    return str(tmp_path)
