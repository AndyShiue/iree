// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/query_tile_sizes_internal.h"

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_riscv_64_f32f32f32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
  // Default tile sizes for RISC-V f32 matrix multiplication
  // These can be optimized based on RISC-V vector extension capabilities
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
}

static iree_uk_matmul_tile_sizes_t
iree_uk_query_matmul_tile_sizes_riscv_64_i8i8i32(
    const iree_uk_query_tile_sizes_2d_params_t* params) {
  // Default tile sizes for RISC-V i8 matrix multiplication
  // These can be optimized based on RISC-V vector extension capabilities
  return (iree_uk_matmul_tile_sizes_t){.M = 8, .K = 1, .N = 8};
}

bool iree_uk_query_matmul_tile_sizes_arch(
    const iree_uk_query_tile_sizes_2d_params_t* params,
    iree_uk_matmul_tile_sizes_t* out_matmul_tile_sizes) {
  iree_uk_uint32_t op = iree_uk_query_tile_sizes_operation(params->flags);
  if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_F32F32F32) {
    *out_matmul_tile_sizes =
        iree_uk_query_matmul_tile_sizes_riscv_64_f32f32f32(params);
    return true;
  } else if (op == IREE_UK_FLAG_QUERY_TILE_SIZES_OPERATION_MATMUL_I8I8I32) {
    *out_matmul_tile_sizes =
        iree_uk_query_matmul_tile_sizes_riscv_64_i8i8i32(params);
    return true;
  } else {
    // Shouldn't happen, validated earlier.
    return false;
  }
} 