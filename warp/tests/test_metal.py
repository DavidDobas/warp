# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior specific to the Metal backend, exercised through the public API."""

import gc
import unittest

import numpy as np

import warp as wp
import warp.sparse
from warp.tests.unittest_utils import StdOutCapture


def metal_available() -> bool:
    return getattr(wp, "is_metal_available", lambda: False)()


@wp.kernel
def increment_kernel(a: wp.array[float]):
    i = wp.tid()
    a[i] = a[i] + 1.0


@wp.kernel
def scale_kernel(a: wp.array[float], s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.kernel
def saxpy_kernel(x: wp.array[float], y: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = 2.0 * x[i] + y[i]


@wp.struct
class ArrayHolder:
    values: wp.array[float]
    offset: float


@wp.kernel
def holder_sum_kernel(holders: wp.array[ArrayHolder], out: wp.array[float]):
    i = wp.tid()
    h = holders[i]
    out[i] = h.values[0] + h.values[1] + h.offset


@wp.kernel
def print_first_kernel():
    print("metal first string")


@wp.kernel
def print_second_kernel():
    print("metal second string")


@unittest.skipUnless(metal_available(), "Requires an Apple GPU")
class TestMetal(unittest.TestCase):
    device = "metal:0"

    def test_host_memory_prefix_then_whole(self):
        """A host range that starts inside an imported range but extends past it is mapped completely."""
        page = 16384
        n = (3 * page) // 4  # three pages of floats: the prefix import covers only the first page
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        prefix = a[:4]
        wp.launch(increment_kernel, dim=4, inputs=[prefix], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.ones(n, dtype=np.float32)
        expected[:4] = 2.0
        np.testing.assert_array_equal(data, expected)

        # releasing the prefix must not take the mapping of the whole array with it
        del prefix
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        np.testing.assert_array_equal(data, expected + 1.0)

    def test_host_memory_inner_page_then_whole(self):
        """An import in the middle of a larger host array does not shadow the rest of that array."""
        page = 16384
        n = (4 * page) // 4
        data = np.zeros(n, dtype=np.float32)
        a = wp.array(data, dtype=float, device="cpu", copy=False)
        first = page // 4 + 8
        inner = a[first : first + 4]  # lies in the second page only
        wp.launch(increment_kernel, dim=4, inputs=[inner], device=self.device)
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        del inner
        gc.collect()
        wp.launch(increment_kernel, dim=n, inputs=[a], device=self.device)
        expected = np.full(n, 2.0, dtype=np.float32)
        expected[first : first + 4] = 3.0
        np.testing.assert_array_equal(data, expected)

    def test_bsr_topology_inside_capture_raises(self):
        """Host-side BSR operations cannot be replayed by a Metal graph, so they refuse to run in a capture."""
        rows = wp.array([0, 1], dtype=int, device=self.device)
        cols = wp.array([0, 1], dtype=int, device=self.device)
        vals = wp.array([1.0, 2.0], dtype=float, device=self.device)
        m = wp.sparse.bsr_zeros(2, 2, block_type=float, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "inside a graph capture"):
            with wp.ScopedCapture(device=self.device):
                wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.sparse.bsr_set_from_triplets(m, rows, cols, vals)  # fine outside a capture
        self.assertEqual(m.nnz_sync(), 2)

    def test_scalar_arguments_are_not_imported(self):
        """NumPy scalars passed by value are not treated as host arrays."""
        device = wp.get_device(self.device)
        a = wp.ones(8, dtype=float, device=self.device)
        wp.launch(scale_kernel, dim=8, inputs=[a, np.float32(3.0)], device=self.device)
        self.assertFalse(device.__dict__.get("_metal_host_memory_pending", False))
        np.testing.assert_array_equal(a.numpy(), np.full(8, 3.0, dtype=np.float32))

    def test_graph_keeps_capture_allocations(self):
        """Arrays allocated during a capture stay valid for replays after Python released them."""
        n = 1024
        x = wp.array(np.arange(n, dtype=np.float32), device=self.device)
        out = wp.zeros(n, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            y = wp.ones(n, dtype=float, device=self.device)  # lives only inside the capture
            wp.launch(saxpy_kernel, dim=n, inputs=[x, y, out], device=self.device)
        del y
        # reuse whatever memory a released buffer would hand back
        junk = [wp.full(n, 7.0, dtype=float, device=self.device) for _ in range(8)]
        out.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(out.numpy(), 2.0 * np.arange(n, dtype=np.float32) + 1.0)
        del junk

    def test_capture_survives_failing_branch(self):
        """A branch body that raises leaves the device usable for normal launches and new captures."""
        cond = wp.ones(1, dtype=int, device=self.device)
        a = wp.zeros(4, dtype=float, device=self.device)

        def failing_body():
            raise ValueError("branch body failed")

        with self.assertRaises(ValueError):
            with wp.ScopedCapture(device=self.device):
                wp.capture_if(cond, on_true=failing_body)
        self.assertFalse(wp.get_device(self.device).is_capturing)
        wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(increment_kernel, dim=4, inputs=[a], device=self.device)
        wp.capture_launch(capture.graph)
        # one eager launch plus one replay: launches inside a capture are recorded, not executed
        np.testing.assert_array_equal(a.numpy(), np.full(4, 2.0, dtype=np.float32))

    def test_replay_translates_nested_arrays_after_new_allocations(self):
        """Graph replays resolve array descriptors stored in device memory after the allocation set changed."""
        values = [wp.array([1.0 + i, 10.0 * (i + 1)], dtype=float, device=self.device) for i in range(4)]
        items = []
        for i, v in enumerate(values):
            h = ArrayHolder()
            h.values = v
            h.offset = float(i)
            items.append(h)
        holders = wp.array(items, dtype=ArrayHolder, device=self.device)
        out = wp.zeros(4, dtype=float, device=self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            wp.launch(holder_sum_kernel, dim=4, inputs=[holders, out], device=self.device)
        extra = [wp.zeros(4096, dtype=float, device=self.device) for _ in range(4)]  # changes the address table
        out.zero_()
        wp.capture_launch(capture.graph)
        expected = np.array([1.0 + i + 10.0 * (i + 1) + i for i in range(4)], dtype=np.float32)
        np.testing.assert_array_equal(out.numpy(), expected)
        del extra

    def test_print_strings_are_per_kernel(self):
        """String constants with the same generated name in different kernels print their own text."""
        capture = StdOutCapture()
        capture.begin()
        wp.launch(print_first_kernel, dim=1, inputs=[], device=self.device)
        wp.launch(print_second_kernel, dim=1, inputs=[], device=self.device)
        wp.synchronize_device(self.device)
        output = capture.end()
        self.assertIn("metal first string", output)
        self.assertIn("metal second string", output)

    def test_to_torch_attaches_gradient(self):
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("Requires PyTorch")
        a = wp.ones(4, dtype=float, device=self.device, requires_grad=True)
        a.grad.fill_(2.0)
        t = wp.to_torch(a)
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.data_ptr(), a.grad.ptr)
        np.testing.assert_array_equal(t.grad.numpy(), np.full(4, 2.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
