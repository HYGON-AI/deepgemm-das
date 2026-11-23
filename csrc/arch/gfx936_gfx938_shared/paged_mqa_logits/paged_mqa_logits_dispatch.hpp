// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#pragma once

#include <torch/extension.h>

#define DISPATCH_MULTI_NEXT_N(next_n, NEXT_N, ...)                  \
    [&]() {                                                         \
        if (next_n == 1) {                                          \
            constexpr int NEXT_N = 1;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 2) {                                   \
            constexpr int NEXT_N = 2;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 3) {                                   \
            constexpr int NEXT_N = 3;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 4) {                                   \
            constexpr int NEXT_N = 4;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 5) {                                   \
            constexpr int NEXT_N = 5;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported next_n size\n");      \
    }()

#define DISPATCH_NEXT_N(next_n, NEXT_N, ...)                        \
    [&]() {                                                         \
        if (next_n == 1) {                                          \
            constexpr int NEXT_N = 1;                               \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (next_n == 2) {                                   \
            constexpr int NEXT_N = 2;                               \
            grid.x *= next_n;                                       \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported next_n size\n");      \
    }()

#define DISPATCH_NUM_HEADS(num_heads, NUM_HEADS, ...)               \
    [&]() {                                                         \
        if (num_heads == 8) {                                       \
            constexpr int NUM_HEADS = 8;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 16) {                               \
            constexpr int NUM_HEADS = 16;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 32) {                               \
            constexpr int NUM_HEADS = 32;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (num_heads == 64) {                               \
            constexpr int NUM_HEADS = 64;                           \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported head size\n");        \
    }()

#define DISPATCH_BLOCK_KV(block_kv, BLOCK_KV, ...)                  \
    [&]() {                                                         \
        if (block_kv == 4) {                                        \
            constexpr int BLOCK_KV = 4;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 8) {                                 \
            constexpr int BLOCK_KV = 8;                             \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 16) {                                \
            constexpr int BLOCK_KV = 16;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 32) {                                \
            constexpr int BLOCK_KV = 32;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        } else if (block_kv == 64) {                                \
            constexpr int BLOCK_KV = 64;                            \
            __VA_ARGS__();                                          \
            return;                                                 \
        }                                                           \
        TORCH_CHECK(false, "unsupported block_kv size\n");    \
    }()
