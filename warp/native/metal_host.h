// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Host-side helpers for structures that are built on the host (meshes, BVHs) and read by Metal
// kernels. The host code builds them into Metal memory and mirrors them in a descriptor whose
// pointers are GPU addresses; the descriptor's own GPU address is the object's id.

#include "mesh.h"
#include "metal.h"
#include "warp.h"

namespace wp {

// Metal device contexts handed to the wp_*_device() entry points are the device ordinal + 1
// (see Device._init_metal in context.py), so that 0 still means "no context".
inline int metal_context_ordinal(void* context) { return int(reinterpret_cast<intptr_t>(context)) - 1; }

// Routes wp_alloc_host()/wp_free_host() on this thread to Metal memory for the scope's lifetime.
struct ScopedMetalHostAlloc {
    int previous;
    explicit ScopedMetalHostAlloc(int ordinal) : previous(wp_host_alloc_redirect_metal(ordinal)) { }
    ~ScopedMetalHostAlloc() { wp_host_alloc_redirect_metal(previous); }
};

// A host-built object mirrored by a descriptor in Metal memory whose GPU address is its id.
template <typename T> struct MetalMirror {
    int ordinal;
    uint64_t host_id;
    T* descriptor;
};

// Rewrites host pointers into Metal allocations as GPU addresses; ok turns false on an unknown pointer.
struct MetalAddressTranslator {
    int ordinal;
    bool ok = true;

    template <typename T> T* operator()(T* host)
    {
        if (!host)
            return nullptr;
        uint64_t address = wp_metal_gpu_address(ordinal, host);
        ok = ok && address != 0;
        return reinterpret_cast<T*>(address);
    }
    template <typename T> array_t<T> operator()(array_t<T> a)
    {
        a.data = (*this)(a.data);
        a.grad = (*this)(a.grad);
        return a;
    }
};

inline BVH metal_bvh_descriptor(MetalAddressTranslator& gpu, const BVH& h)
{
    BVH d = h;
    d.node_lowers = gpu(h.node_lowers);
    d.node_uppers = gpu(h.node_uppers);
    d.node_parents = gpu(h.node_parents);
    d.node_counts = gpu(h.node_counts);
    d.primitive_indices = gpu(h.primitive_indices);
    d.root = gpu(h.root);
    d.item_lowers = gpu(h.item_lowers);
    d.item_uppers = gpu(h.item_uppers);
    d.item_groups = gpu(h.item_groups);
    return d;
}

inline Mesh metal_mesh_descriptor(MetalAddressTranslator& gpu, const Mesh& h)
{
    Mesh d = h;
    d.points = gpu(h.points);
    d.velocities = gpu(h.velocities);
    d.indices = gpu(h.indices);
    d.lowers = gpu(h.lowers);
    d.uppers = gpu(h.uppers);
    d.solid_angle_props = gpu(h.solid_angle_props);
    d.bvh = metal_bvh_descriptor(gpu, h.bvh);
    return d;
}

}  // namespace wp
