# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vectors scanned and modified inside a function, then returned (MJWarp's weight accumulator).

The Metal compiler miscompiled this pattern when the scan read elements through dynamic
indexing; see vec_get_at in vec.h.
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

vec16i = wp.types.vector(16, wp.int32)
vec16f = wp.types.vector(16, wp.float32)


@wp.func
def add_weight(nb: int, body: vec16i, weight: vec16f, b: int, w: float) -> tuple[int, vec16i, vec16f]:
    for i in range(16):
        if i >= nb:
            break
        if body[i] == b:
            weight[i] += w
            return nb, body, weight
    if nb < 16:
        body[nb] = b
        weight[nb] = w
        return nb + 1, body, weight
    return nb, body, weight


@wp.kernel
def accumulate_kernel(ids: wp.array2d[int], nb_out: wp.array[int], body_out: wp.array[vec16i], weight_out: wp.array[vec16f]):
    tid = wp.tid()
    body = vec16i(-1)
    weight = vec16f(0.0)
    nb = int(0)
    for j in range(12):
        nb, body, weight = add_weight(nb, body, weight, ids[tid, j], 1.0)
    nb_out[tid] = nb
    body_out[tid] = body
    weight_out[tid] = weight


def test_vec_scan(test, device):
    rng = np.random.default_rng(3)
    ids_np = rng.integers(1, 10, size=(64, 12)).astype(np.int32)
    ids = wp.array(ids_np, dtype=int, device=device)
    nb = wp.zeros(64, dtype=int, device=device)
    body = wp.zeros(64, dtype=vec16i, device=device)
    weight = wp.zeros(64, dtype=vec16f, device=device)
    wp.launch(accumulate_kernel, dim=64, inputs=[ids, nb, body, weight], device=device)

    for row, expected_ids in enumerate(ids_np):
        unique, counts = np.unique(expected_ids, return_counts=True)
        order = np.argsort([list(expected_ids).index(u) for u in unique])  # first-seen order
        test.assertEqual(nb.numpy()[row], len(unique))
        assert_np_equal(body.numpy()[row][: len(unique)], unique[order])
        assert_np_equal(weight.numpy()[row][: len(unique)], counts[order].astype(np.float32))


devices = get_test_devices()


class TestVecScan(unittest.TestCase):
    pass


add_function_test(TestVecScan, "test_vec_scan", test_vec_scan, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
