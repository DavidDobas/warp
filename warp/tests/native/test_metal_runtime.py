# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the native Metal runtime (warp/native/metal.mm) driven directly through ctypes.

These tests deliberately bypass the Warp Python runtime so they exercise the C API exactly as the
Python device layer will: shared-storage buffers addressed by host pointer, an argument struct whose
pointer fields are translated to GPU addresses at launch, and the deferred-free/argument-ring machinery.
"""

import ctypes
import os
import platform
import unittest

import numpy as np

LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bin", "libwarp.dylib"
)

MSL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
struct bounds_t { ulong size; int ndim; int shape[4]; };
struct arr_t { device float* data; device float* grad; int shape[4]; int strides[4]; ushort ndim; ushort flags; };
struct args_t { arr_t x; arr_t y; device atomic_int* counter; };
struct big_args_t { arr_t x; arr_t y; device atomic_int* counter; int pad[1024]; };
kernel void saxpy(constant bounds_t& dim [[buffer(0)]], constant args_t* args [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim.size) return;
    args->y.data[tid] = 2.0f * args->x.data[tid] + 1.0f;
    atomic_fetch_add_explicit(args->counter, 1, memory_order_relaxed);
}
kernel void tg_reduce(constant bounds_t& dim [[buffer(0)]], constant args_t* args [[buffer(1)]], uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_threadgroup]], threadgroup int* smem [[threadgroup(0)]]) {
    smem[lane] = tid < dim.size ? int(args->x.data[tid]) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;
    int sum = 0;
    for (uint i = 0; i < 256; ++i) sum += smem[i];
    atomic_fetch_add_explicit(args->counter, sum, memory_order_relaxed);
}
kernel void saxpy_big(constant bounds_t& dim [[buffer(0)]], constant big_args_t* args [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= dim.size) return;
    args->y.data[tid] = 2.0f * args->x.data[tid] + 1.0f + float(args->pad[1023]);
    atomic_fetch_add_explicit(args->counter, 1, memory_order_relaxed);
}
"""


class bounds_t(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("ndim", ctypes.c_int32), ("shape", ctypes.c_int32 * 4)]


class arr_t(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("grad", ctypes.c_void_p),
        ("shape", ctypes.c_int32 * 4),
        ("strides", ctypes.c_int32 * 4),
        ("ndim", ctypes.c_uint16),
        ("flags", ctypes.c_uint16),
    ]


class args_t(ctypes.Structure):
    _fields_ = [("x", arr_t), ("y", arr_t), ("counter", ctypes.c_void_p)]


class big_args_t(ctypes.Structure):
    _fields_ = [("x", arr_t), ("y", arr_t), ("counter", ctypes.c_void_p), ("pad", ctypes.c_int32 * 1024)]


def pointer_offsets(args_type):
    """Byte offsets of every pointer field in an args struct, as the Python layer must supply them."""
    offsets = [args_type.x.offset + arr_t.data.offset, args_type.x.offset + arr_t.grad.offset]
    offsets += [args_type.y.offset + arr_t.data.offset, args_type.y.offset + arr_t.grad.offset]
    offsets.append(args_type.counter.offset)
    return (ctypes.c_size_t * len(offsets))(*offsets)


def load_library():
    lib = ctypes.CDLL(LIB_PATH)
    lib.wp_get_error_string.restype = ctypes.c_char_p
    lib.wp_set_error_output_enabled.argtypes = [ctypes.c_int]
    lib.wp_is_metal_enabled.restype = ctypes.c_int
    lib.wp_metal_device_get_count.restype = ctypes.c_int
    lib.wp_metal_device_get_name.argtypes = [ctypes.c_int]
    lib.wp_metal_device_get_name.restype = ctypes.c_char_p
    lib.wp_metal_device_get_max_threadgroup_memory.argtypes = [ctypes.c_int]
    lib.wp_metal_device_get_max_threads_per_threadgroup.argtypes = [ctypes.c_int]
    lib.wp_metal_device_has_unified_memory.argtypes = [ctypes.c_int]
    lib.wp_alloc_metal.argtypes = [ctypes.c_int, ctypes.c_size_t]
    lib.wp_alloc_metal.restype = ctypes.c_void_p
    lib.wp_free_metal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    lib.wp_metal_load_library.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.wp_metal_load_library.restype = ctypes.c_void_p
    lib.wp_metal_unload_library.argtypes = [ctypes.c_void_p]
    lib.wp_metal_get_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.wp_metal_get_kernel.restype = ctypes.c_void_p
    lib.wp_metal_get_kernel_thread_execution_width.argtypes = [ctypes.c_void_p]
    lib.wp_metal_get_kernel_max_threads_per_threadgroup.argtypes = [ctypes.c_void_p]
    lib.wp_metal_launch_kernel.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
        ctypes.c_size_t,
    ]
    lib.wp_metal_launch_kernel.restype = ctypes.c_int
    lib.wp_metal_synchronize.argtypes = [ctypes.c_int]
    lib.wp_metal_synchronize.restype = ctypes.c_int
    lib.wp_metal_flush.argtypes = [ctypes.c_int]
    lib.wp_metal_flush.restype = ctypes.c_int
    return lib


def metal_available():
    if platform.system() != "Darwin" or not os.path.exists(LIB_PATH):
        return False
    lib = load_library()
    return bool(lib.wp_is_metal_enabled()) and lib.wp_metal_device_get_count() > 0


@unittest.skipUnless(metal_available(), "Requires an Apple GPU and a Metal-enabled libwarp")
class TestMetalRuntime(unittest.TestCase):
    DEVICE = 0
    N = 1000

    @classmethod
    def setUpClass(cls):
        cls.lib = load_library()
        cls.library = cls.lib.wp_metal_load_library(cls.DEVICE, MSL_SOURCE.encode())
        assert cls.library, cls.lib.wp_get_error_string()
        cls.saxpy = cls.lib.wp_metal_get_kernel(cls.library, b"saxpy")
        cls.saxpy_big = cls.lib.wp_metal_get_kernel(cls.library, b"saxpy_big")
        cls.tg_reduce = cls.lib.wp_metal_get_kernel(cls.library, b"tg_reduce")
        assert cls.saxpy and cls.saxpy_big and cls.tg_reduce, cls.lib.wp_get_error_string()

    @classmethod
    def tearDownClass(cls):
        cls.lib.wp_metal_synchronize(cls.DEVICE)
        cls.lib.wp_metal_unload_library(cls.library)

    def error_string(self):
        return self.lib.wp_get_error_string().decode()

    def alloc_floats(self, n, values=None):
        ptr = self.lib.wp_alloc_metal(self.DEVICE, n * 4)
        self.assertTrue(ptr, self.error_string())
        view = np.ctypeslib.as_array((ctypes.c_float * n).from_address(ptr))
        view[:] = 0.0 if values is None else values
        return ptr, view

    def alloc_counter(self):
        ptr = self.lib.wp_alloc_metal(self.DEVICE, 4)
        self.assertTrue(ptr, self.error_string())
        counter = ctypes.c_int32.from_address(ptr)
        counter.value = 0
        return ptr, counter

    def free(self, *ptrs):
        for ptr in ptrs:
            self.lib.wp_free_metal(self.DEVICE, ptr)

    def make_args(self, x_ptr, y_ptr, counter_ptr, args_type=args_t):
        args = args_type()
        for arr, ptr in ((args.x, x_ptr), (args.y, y_ptr)):
            arr.data = ptr
            arr.grad = None
            arr.shape[0] = self.N
            arr.strides[0] = 4
            arr.ndim = 1
        args.counter = counter_ptr
        return args

    def launch(self, kernel, args, threads_per_threadgroup=256, num_threads=None, threadgroup_memory_bytes=0):
        bounds = bounds_t(size=self.N if num_threads is None else num_threads, ndim=1)
        bounds.shape[0] = bounds.size
        offsets = pointer_offsets(type(args))
        return self.lib.wp_metal_launch_kernel(
            self.DEVICE,
            kernel,
            bounds.size,
            threads_per_threadgroup,
            ctypes.byref(bounds),
            ctypes.sizeof(bounds),
            ctypes.byref(args),
            ctypes.sizeof(args),
            offsets,
            len(offsets),
            threadgroup_memory_bytes,
        )

    def synchronize(self):
        self.assertEqual(self.lib.wp_metal_synchronize(self.DEVICE), 0, self.error_string())

    def test_device_queries(self):
        lib = self.lib
        self.assertTrue(lib.wp_metal_device_get_name(self.DEVICE))
        self.assertGreaterEqual(lib.wp_metal_device_get_max_threadgroup_memory(self.DEVICE), 16 * 1024)
        self.assertGreaterEqual(lib.wp_metal_device_get_max_threads_per_threadgroup(self.DEVICE), 256)
        self.assertEqual(lib.wp_metal_device_has_unified_memory(self.DEVICE), 1)
        self.assertGreaterEqual(lib.wp_metal_get_kernel_thread_execution_width(self.saxpy), 1)
        self.assertGreaterEqual(lib.wp_metal_get_kernel_max_threads_per_threadgroup(self.saxpy), 256)

        lib.wp_set_error_output_enabled(0)
        try:
            self.assertIsNone(lib.wp_metal_device_get_name(lib.wp_metal_device_get_count()))
            self.assertIn("ordinal", self.error_string())
        finally:
            lib.wp_set_error_output_enabled(1)

    def test_struct_layout_matches_msl(self):
        # The MSL structs follow C layout rules; ctypes must agree for the offsets to be meaningful.
        self.assertEqual(ctypes.sizeof(bounds_t), 32)
        self.assertEqual(ctypes.sizeof(arr_t), 56)
        self.assertEqual(ctypes.sizeof(args_t), 120)
        self.assertGreater(ctypes.sizeof(big_args_t), 4096)

    def test_saxpy(self):
        x = np.arange(self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(self.N, x)
        y_ptr, y = self.alloc_floats(self.N)
        counter_ptr, counter = self.alloc_counter()

        args = self.make_args(x_ptr, y_ptr, counter_ptr)
        self.assertEqual(self.launch(self.saxpy, args), 0, self.error_string())
        self.synchronize()

        np.testing.assert_allclose(y, 2.0 * x + 1.0)
        self.assertEqual(counter.value, self.N)
        self.free(x_ptr, y_ptr, counter_ptr)

    def test_large_args(self):
        # Argument structs above the 4 KB setBytes limit must go through the argument ring buffer.
        x = np.arange(self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(self.N, x)
        y_ptr, y = self.alloc_floats(self.N)
        counter_ptr, counter = self.alloc_counter()

        args = self.make_args(x_ptr, y_ptr, counter_ptr, big_args_t)
        args.pad[1023] = 5
        self.assertEqual(self.launch(self.saxpy_big, args, threads_per_threadgroup=1 << 20), 0, self.error_string())
        self.synchronize()

        np.testing.assert_allclose(y, 2.0 * x + 6.0)
        self.assertEqual(counter.value, self.N)
        self.free(x_ptr, y_ptr, counter_ptr)

    def test_threadgroup_memory(self):
        # With threadgroup memory, whole threadgroups are dispatched: lanes beyond the bounds still run.
        x = np.arange(self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(self.N, x)
        y_ptr, _ = self.alloc_floats(self.N)
        counter_ptr, counter = self.alloc_counter()

        args = self.make_args(x_ptr, y_ptr, counter_ptr)
        self.assertEqual(self.launch(self.tg_reduce, args, threadgroup_memory_bytes=256 * 4), 0, self.error_string())
        self.synchronize()

        self.assertEqual(counter.value, int(x.sum()))
        self.free(x_ptr, y_ptr, counter_ptr)

    def test_offset_pointers_and_null_pointers(self):
        # Pointers into the middle of an allocation resolve to the same offset on the GPU.
        x = np.arange(2 * self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(2 * self.N, x)
        y_ptr, y = self.alloc_floats(2 * self.N)
        counter_ptr, counter = self.alloc_counter()

        args = self.make_args(x_ptr + self.N * 4, y_ptr + self.N * 4, counter_ptr)
        self.assertEqual(self.launch(self.saxpy, args), 0, self.error_string())
        self.synchronize()

        np.testing.assert_allclose(y[: self.N], 0.0)
        np.testing.assert_allclose(y[self.N :], 2.0 * x[self.N :] + 1.0)
        self.assertEqual(counter.value, self.N)

        # Zero-thread launches are a no-op.
        self.assertEqual(self.launch(self.saxpy, args, num_threads=0), 0, self.error_string())
        self.synchronize()
        self.assertEqual(counter.value, self.N)
        self.free(x_ptr, y_ptr, counter_ptr)

    def test_invalid_pointer_is_rejected(self):
        y_ptr, _ = self.alloc_floats(self.N)
        counter_ptr, _ = self.alloc_counter()
        host_array = np.zeros(self.N, dtype=np.float32)
        args = self.make_args(host_array.ctypes.data, y_ptr, counter_ptr)

        self.lib.wp_set_error_output_enabled(0)
        try:
            self.assertNotEqual(self.launch(self.saxpy, args), 0)
        finally:
            self.lib.wp_set_error_output_enabled(1)
        message = self.error_string()
        self.assertIn(f"offset {args_t.x.offset + arr_t.data.offset}", message)
        self.assertIn("not Metal memory", message)

        # Freed memory must be rejected as well.
        self.free(y_ptr)
        args = self.make_args(counter_ptr, y_ptr, counter_ptr)
        self.lib.wp_set_error_output_enabled(0)
        try:
            self.assertNotEqual(self.launch(self.saxpy, args), 0)
        finally:
            self.lib.wp_set_error_output_enabled(1)
        self.assertIn(f"offset {args_t.y.offset + arr_t.data.offset}", self.error_string())
        self.free(counter_ptr)

    def test_compile_errors(self):
        self.lib.wp_set_error_output_enabled(0)
        try:
            self.assertIsNone(self.lib.wp_metal_load_library(self.DEVICE, b"kernel void broken( { }"))
            self.assertIn("error", self.error_string())
            self.assertIsNone(self.lib.wp_metal_get_kernel(self.library, b"missing"))
            self.assertIn("missing", self.error_string())
        finally:
            self.lib.wp_set_error_output_enabled(1)

    def test_free_while_launch_pending(self):
        x = np.arange(self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(self.N, x)
        y_ptr, y = self.alloc_floats(self.N)
        counter_ptr, _ = self.alloc_counter()

        args = self.make_args(x_ptr, y_ptr, counter_ptr)
        self.assertEqual(self.launch(self.saxpy, args), 0, self.error_string())
        self.free(x_ptr)  # GPU may still read x; the release must be deferred
        self.assertEqual(self.lib.wp_metal_flush(self.DEVICE), 0, self.error_string())
        self.free(counter_ptr)
        self.synchronize()

        np.testing.assert_allclose(y, 2.0 * x + 1.0)
        self.free(y_ptr)

    def test_many_launches(self):
        # 3000 launches with 256-byte argument slots wrap the 16 MB argument ring and trigger auto-commits.
        num_launches = 3000
        x = np.arange(self.N, dtype=np.float32)
        x_ptr, _ = self.alloc_floats(self.N, x)
        y_ptr, y = self.alloc_floats(self.N)
        counter_ptr, counter = self.alloc_counter()

        args = self.make_args(x_ptr, y_ptr, counter_ptr)
        for _ in range(num_launches):
            self.assertEqual(self.launch(self.saxpy, args), 0, self.error_string())
        self.synchronize()

        np.testing.assert_allclose(y, 2.0 * x + 1.0)
        self.assertEqual(counter.value, num_launches * self.N)

        # Enough launches with large arguments to wrap the ring several times without an explicit synchronize.
        big_args = self.make_args(x_ptr, y_ptr, counter_ptr, big_args_t)
        counter.value = 0
        for _ in range(num_launches):
            self.assertEqual(self.launch(self.saxpy_big, big_args), 0, self.error_string())
        self.synchronize()
        self.assertEqual(counter.value, num_launches * self.N)
        self.free(x_ptr, y_ptr, counter_ptr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
