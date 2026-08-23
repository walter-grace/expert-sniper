"""Binary wire format between driver and expert nodes.

Request:
  [layer_idx:u16][n_experts:u16][expert_ids:u16*N]
  [hs_shape:u16*3][inds_shape:u16*3][wt_shape:u16*3]
  [hidden:f16][indices:i32][weights:f32]
Response:
  [n_computed:u16][ndim:u16][shape:u16*ndim][result:f16]
"""
import struct

import numpy as np


def pack_request(layer_idx, expert_ids, hidden_f16, inds_i32, weights_f32):
    ids = np.asarray(sorted(expert_ids), dtype=np.uint16)
    parts = [
        struct.pack("<HH", layer_idx, len(ids)),
        ids.tobytes(),
        np.asarray(hidden_f16.shape, dtype=np.uint16).tobytes(),
        np.asarray(inds_i32.shape, dtype=np.uint16).tobytes(),
        np.asarray(weights_f32.shape, dtype=np.uint16).tobytes(),
        hidden_f16.tobytes(),
        inds_i32.tobytes(),
        weights_f32.tobytes(),
    ]
    return b"".join(parts)


def unpack_request(raw):
    layer_idx, n_experts = struct.unpack("<HH", raw[:4])
    off = 4
    expert_ids = np.frombuffer(raw[off:off + n_experts * 2], dtype=np.uint16)
    off += n_experts * 2
    shapes = []
    for _ in range(3):
        shapes.append(tuple(int(x) for x in
                            np.frombuffer(raw[off:off + 6], dtype=np.uint16)))
        off += 6
    hs_shape, inds_shape, wt_shape = shapes
    n = int(np.prod(hs_shape)) * 2
    hidden = np.frombuffer(raw[off:off + n], dtype=np.float16).copy().reshape(hs_shape)
    off += n
    n = int(np.prod(inds_shape)) * 4
    inds = np.frombuffer(raw[off:off + n], dtype=np.int32).copy().reshape(inds_shape)
    off += n
    n = int(np.prod(wt_shape)) * 4
    weights = np.frombuffer(raw[off:off + n], dtype=np.float32).copy().reshape(wt_shape)
    return layer_idx, [int(e) for e in expert_ids], hidden, inds, weights


def pack_response(n_computed, result_f16):
    hdr = struct.pack("<HH", n_computed, result_f16.ndim)
    shape = np.asarray(result_f16.shape, dtype=np.uint16).tobytes()
    return hdr + shape + result_f16.tobytes()


def unpack_response(raw):
    n_computed, ndim = struct.unpack("<HH", raw[:4])
    off = 4
    shape = tuple(int(x) for x in
                  np.frombuffer(raw[off:off + ndim * 2], dtype=np.uint16))
    off += ndim * 2
    result = np.frombuffer(raw[off:], dtype=np.float16).reshape(shape)
    return n_computed, result
