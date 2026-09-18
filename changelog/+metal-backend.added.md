Add a Metal backend for Apple GPUs: kernels, tiles, backward kernels, graph capture, meshes, hash grids, volumes
and textures run on a `metal:0` device on Apple Silicon. Arrays live in unified memory, NumPy arrays and
Torch CPU tensors can be passed to Metal kernels in place, and `wp.to_torch()` returns a zero-copy CPU tensor.
`float64`, `wp.fixedarray`, fabric arrays, deterministic mode, saveable captures and tile-kernel adjoints
are not supported on Metal.
The hash of every module now includes a digest of the native headers, so cached kernels are rebuilt once after
upgrading and whenever the headers change; ahead-of-time module names depend on the header contents as well.
