// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/builtins/ukernel/mmt4d_internal.h"

iree_uk_mmt4d_tile_func_t iree_uk_mmt4d_select_tile_func_arch(
    const iree_uk_mmt4d_params_t* params) {
  // For now, RISC-V implementation returns null, falling back to generic
  // implementation. This can be extended with RISC-V specific optimizations
  // using RISC-V vector extensions (RVV) or other RISC-V specific features.
  return 0;
}