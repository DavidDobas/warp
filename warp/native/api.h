// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Export macro for headers that do not include crt.h (same definition as there).

#ifndef WP_API
#if !defined(__CUDA_ARCH__)
#if defined(_WIN32)
#define WP_API __declspec(dllexport)
#else
#define WP_API __attribute__((visibility("default")))
#endif
#else
#define WP_API
#endif
#endif  // WP_API
