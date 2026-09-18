Apple GPUs (Metal)
==================

.. currentmodule:: warp

Warp can run kernels on Apple Silicon GPUs through a Metal backend. On a Mac with a Metal-enabled build,
the GPU appears as the device ``"metal:0"`` and is the default device:

.. code-block:: python

    import warp as wp


    @wp.kernel
    def scale(a: wp.array[float], s: float):
        i = wp.tid()
        a[i] = a[i] * s


    a = wp.array([1.0, 2.0, 3.0], dtype=float, device="metal:0")
    wp.launch(scale, dim=3, inputs=[a, 2.0], device="metal:0")
    print(a.numpy())  # [2. 4. 6.]

:func:`wp.is_metal_available() <warp.is_metal_available>` reports whether a Metal device was found, and
``Device.is_metal`` identifies one. Code that selects a device with ``wp.get_device()``
or ``wp.ScopedDevice`` needs no changes. Code that tests ``device.is_cuda`` to decide whether it runs on a GPU
should test ``not device.is_cpu`` instead.

Requirements
------------

* Apple Silicon and macOS 15 or newer.
* A build of Warp that contains the Metal runtime. The macOS wheels on PyPI do not. Build from source with
  ``build_lib.py`` or CMake (see :ref:`building-from-source`); both compile ``warp/native/metal.mm`` on macOS.
  Xcode is not needed at run time: kernels are compiled from source by the Metal framework when a module loads,
  and the result is kept in the kernel cache.

Supported Features
------------------

Kernels and user functions, structs, tile operations, backward kernels and :class:`wp.Tape <Tape>`,
graph capture and replay including :func:`wp.capture_if() <capture_if>` and :func:`wp.capture_while() <capture_while>`,
meshes, BVHs, hash grids, NanoVDB volumes, textures, atomic operations, ``wp.ref`` parameters,
``print()`` and assertions inside kernels, and DLPack, NumPy and PyTorch interoperability.

Memory Model
------------

Apple GPUs share memory with the CPU. Arrays on a Metal device live in unified memory, so their ``ptr`` is a valid
host pointer:

* ``array.numpy()`` and ``__array_interface__`` return a view, not a copy. Warp synchronizes the device before
  handing out the pointer.
* :func:`wp.to_torch() <warp.to_torch>` returns a zero-copy PyTorch **CPU** tensor, with ``grad`` attached when the
  array has one. PyTorch's ``mps`` tensors cannot alias Warp arrays: PyTorch does not expose their buffers, and it
  schedules work on its own command queue. Move data to ``mps`` explicitly when a network should run on the GPU.
* NumPy arrays and PyTorch CPU tensors can be passed to a Metal kernel directly. Warp maps their pages into the
  GPU address space for the duration of the array's lifetime, without copying, and synchronizes after such a launch.
* Copies between ``"cpu"`` and ``"metal:0"`` are plain memory copies.

Limitations
-----------

* **No** ``float64``. Apple GPUs have no double-precision type. A kernel that uses ``wp.float64``, or a vector,
  matrix or struct built on it, raises when it is launched on Metal. The rest of its module still loads.
* **Tile kernels are forward only.** Kernels that use tile operations have no adjoint on Metal.
* If the adjoint of a module fails to compile on Metal, Warp warns and rebuilds the module forward-only, so the
  forward pass keeps working. Launching one of its backward kernels raises.
* Tile kernels are limited by threadgroup memory, 32 KB on current Apple GPUs. A kernel whose tiles need more
  raises at launch. Keeping block-local tiles small, or accumulating into an output tile
  (``wp.tile_matmul(a, b, out)``) instead of creating temporaries, stays below the limit.
* Signed 64-bit ``wp.atomic_min()`` and ``wp.atomic_max()`` are read-modify-write and not atomic.
* Spinlocks built from ``wp.atomic_cas()`` can hang: Apple GPUs do not guarantee forward progress between
  SIMD lanes.
* ``wp.fixedarray``, fabric arrays, deterministic mode (``wp.config.deterministic``), saveable (APIC) captures and
  the allocation tracker are not supported.
* Host-side utilities that are not recorded into a graph, such as building or changing the topology of a
  :class:`warp.sparse.BsrMatrix`, raise when called during a graph capture on Metal. Call them outside the
  capture.
* Native snippets (:func:`wp.func_native() <func_native>`) must be valid Metal Shading Language. Pointer casts need
  an address space, for which Warp defines ``WP_THREAD`` and ``WP_DEVICE`` (both empty on CPU and CUDA), and
  ``long long`` is not available.
* Transcendental functions can differ from their CPU results in the last bit.

Performance Notes
-----------------

* Launches are recorded into a command buffer and run asynchronously. Reading an array on the host synchronizes.
* Graph capture removes most of the per-launch cost and is worth using for many small kernels.
* ``wp.capture_while()`` evaluates its condition on the host between iterations on Metal. A fixed iteration count
  is usually faster for short loops.
* Tile kernels run one tile per threadgroup. Their throughput is bounded by threadgroup memory, so they scale less
  well with problem size than they do on CUDA.
