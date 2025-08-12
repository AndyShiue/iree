// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_
#define IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_

#include <riscv_vector.h>

#include "iree/builtins/ukernel/common.h"
#include "iree/schemas/cpu_data.h"

#if defined(IREE_DEVICE_STANDALONE)
// Standalone builds (e.g. bitcode) use our own Clang, supporting everything.
#define IREE_UK_BUILD_RISCV_64_V
#else
// Compiling with the system toolchain. Include the configured header.
#include "iree/builtins/ukernel/arch/riscv_64/config_riscv_64.h"
#endif

static inline bool iree_uk_cpu_riscv_64(const iree_uk_uint64_t* cpu_data) {
  (void)cpu_data;
  return true;
}

/*static inline bool iree_uk_cpu_arm_64_fullfp16(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_FULLFP16);
}

static inline bool iree_uk_cpu_arm_64_fp16fml(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_FP16FML);
}

static inline bool iree_uk_cpu_arm_64_bf16(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_BF16);
}

static inline bool iree_uk_cpu_arm_64_dotprod(
    const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_DOTPROD);
}

static inline bool iree_uk_cpu_arm_64_i8mm(const iree_uk_uint64_t* cpu_data) {
  return iree_uk_all_bits_set(cpu_data[0], IREE_CPU_DATA0_ARM_64_I8MM);
}*/

//[8x4_x8]
static inline vint8m1x2_t iree_uk_load_8x4xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  /*int32x4_t v0_i32 = vdupq_n_s32(0);
  int32x4_t v1_i32 = vdupq_n_s32(0);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 0 * stride), v0_i32, 0);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 1 * stride), v0_i32, 1);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 2 * stride), v0_i32, 2);
  v0_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 3 * stride), v0_i32, 3);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 4 * stride), v1_i32, 0);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 5 * stride), v1_i32, 1);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 6 * stride), v1_i32, 2);
  v1_i32 =
      vld1q_lane_s32((const iree_uk_int32_t*)(src + 7 * stride), v1_i32, 3);
  int8x16x2_t v;
  v.val[0] = vreinterpretq_s8_s32(v0_i32);
  v.val[1] = vreinterpretq_s8_s32(v1_i32);
  return v;*/
  size_t vl = __riscv_vsetvl_e32m1(4);
  vint32m1_t lane0 = __riscv_vle32_v_i32m1((const int32_t*)(src + 0 * stride), vl);
  vint32m1_t lane1 = __riscv_vle32_v_i32m1((const int32_t*)(src + 1 * stride), vl);
  vint32m1_t lane2 = __riscv_vle32_v_i32m1((const int32_t*)(src + 2 * stride), vl);
  vint32m1_t lane3 = __riscv_vle32_v_i32m1((const int32_t*)(src + 3 * stride), vl);
  int32_t e0 = __riscv_vmv_x_s_i32m1_i32(lane0);
  int32_t e1 = __riscv_vmv_x_s_i32m1_i32(lane1);
  int32_t e2 = __riscv_vmv_x_s_i32m1_i32(lane2);
  int32_t e3 = __riscv_vmv_x_s_i32m1_i32(lane3);
  vint32m1_t temp0 = __riscv_vmv_v_x_i32m1(e0, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e1, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e2, vl);
  temp0 = __riscv_vslide1down_vx_i32m1(temp0, e3, vl);
  vint32m1_t lane4 = __riscv_vle32_v_i32m1((const int32_t*)(src + 4 * stride), vl);
  vint32m1_t lane5 = __riscv_vle32_v_i32m1((const int32_t*)(src + 5 * stride), vl);
  vint32m1_t lane6 = __riscv_vle32_v_i32m1((const int32_t*)(src + 6 * stride), vl);
  vint32m1_t lane7 = __riscv_vle32_v_i32m1((const int32_t*)(src + 7 * stride), vl);
  int32_t e4 = __riscv_vmv_x_s_i32m1_i32(lane4);
  int32_t e5 = __riscv_vmv_x_s_i32m1_i32(lane5);
  int32_t e6 = __riscv_vmv_x_s_i32m1_i32(lane6);
  int32_t e7 = __riscv_vmv_x_s_i32m1_i32(lane7);
  vint32m1_t temp1 = __riscv_vmv_v_x_i32m1(e4, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e5, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e6, vl);
  temp1 = __riscv_vslide1down_vx_i32m1(temp1, e7, vl);
  vint8m1_t v0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(temp0);
  vint8m1_t v1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(temp1);
  vint8m1x2_t v = __riscv_vcreate_v_i8m1x2(v0_i8, v1_i8);
  //size_t vl = __riscv_vsetvl_e32m1(4);
  //vint32m1_t dummy_i32 = __riscv_vmv_v_x_i32m1(0, vl);
  //vint8m1_t dummy = __riscv_vreinterpret_v_i32m1_i8m1(dummy_i32);
  //vint8m1x2_t v = __riscv_vcreate_v_i8m1x2(dummy, dummy);
  return v;
}

//[8x4_x8]
static inline vint8m1x4_t iree_uk_load_8x8xi8_strided_permute(
    const iree_uk_int8_t* src, iree_uk_index_t stride, int p0, int p1, int p2,
    int p3, int p4, int p5, int p6, int p7) {
  /*int8x8_t row0 = vld1_s8(src + p0 * stride);
  int8x8_t row1 = vld1_s8(src + p1 * stride);
  int8x8_t row2 = vld1_s8(src + p2 * stride);
  int8x8_t row3 = vld1_s8(src + p3 * stride);
  int8x8_t row4 = vld1_s8(src + p4 * stride);
  int8x8_t row5 = vld1_s8(src + p5 * stride);
  int8x8_t row6 = vld1_s8(src + p6 * stride);
  int8x8_t row7 = vld1_s8(src + p7 * stride);
  int8x16x4_t v;
  v.val[0] = vcombine_s8(row0, row1);
  v.val[1] = vcombine_s8(row2, row3);
  v.val[2] = vcombine_s8(row4, row5);
  v.val[3] = vcombine_s8(row6, row7);
  return v;*/
  size_t vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t row0 = __riscv_vle8_v_i8m1(src + p0 * stride, vl);
  vint8m1_t row1 = __riscv_vle8_v_i8m1(src + p1 * stride, vl);
  vint8m1_t row2 = __riscv_vle8_v_i8m1(src + p2 * stride, vl);
  vint8m1_t row3 = __riscv_vle8_v_i8m1(src + p3 * stride, vl);
  vint8m1_t row4 = __riscv_vle8_v_i8m1(src + p4 * stride, vl);
  vint8m1_t row5 = __riscv_vle8_v_i8m1(src + p5 * stride, vl);
  vint8m1_t row6 = __riscv_vle8_v_i8m1(src + p6 * stride, vl);
  vint8m1_t row7 = __riscv_vle8_v_i8m1(src + p7 * stride, vl);
  vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t v0 = __riscv_vslideup_vx_i8m1(row0, row1, half_vl, vl);
  vint8m1_t v1 = __riscv_vslideup_vx_i8m1(row2, row3, half_vl, vl);
  vint8m1_t v2 = __riscv_vslideup_vx_i8m1(row4, row5, half_vl, vl);
  vint8m1_t v3 = __riscv_vslideup_vx_i8m1(row6, row7, half_vl, vl);
  //size_t vl = __riscv_vsetvl_e8m1(8);
  //vint8m1_t v0 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v1 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v2 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t v3 = __riscv_vmv_v_x_i8m1(0, vl);
  vint8m1x4_t v = __riscv_vcreate_v_i8m1x4(v0, v1, v2, v3);
  return v;
}

static inline vint8m1x4_t iree_uk_load_8x8xi8_strided(
    const iree_uk_int8_t* src, iree_uk_index_t stride) {
  return iree_uk_load_8x8xi8_strided_permute(src, stride, 0, 1, 2, 3, 4, 5,
                                                  6, 7);
}

static inline vint16m1x2_t iree_uk_zip_16xi8_as_8xi16(vint8m1_t a,
                                                          vint8m1_t b) {
  /*int8x16x2_t z = vzipq_s8(a, b);
  int16x8x2_t r;
  r.val[0] = vreinterpretq_s16_s8(z.val[0]);
  r.val[1] = vreinterpretq_s16_s8(z.val[1]);*/
  //z.val[0] = {a0, b0, a1, b1, ..., a7, b7}
  //z.val[1] = {a8, b8, a9, b9, ..., a15, b15}
  size_t vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t a_lo = __riscv_vslidedown_vx_i8m1(a, 0, half_vl);
  vint8m1_t a_hi = __riscv_vslidedown_vx_i8m1(a, half_vl, half_vl);
  vint8m1_t b_lo = __riscv_vslidedown_vx_i8m1(b, 0, half_vl);
  vint8m1_t b_hi = __riscv_vslidedown_vx_i8m1(b, half_vl, half_vl);
  vint8m1_t in0 = __riscv_vslideup_vx_i8m1(a_lo, b_lo, half_vl, vl);
  vint8m1_t in1 = __riscv_vslideup_vx_i8m1(a_hi, b_hi, half_vl, vl);
  uint8_t lane[16] = {0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15};
  vuint8m1_t idx = __riscv_vle8_v_u8m1(lane, vl);
  vint8m1_t z0 = __riscv_vrgather_vv_i8m1(in0, idx, vl);
  vint8m1_t z1 = __riscv_vrgather_vv_i8m1(in1, idx, vl);
  /*int8_t a_raw[16], b_raw[16], temp0[16], temp1[16];
  __riscv_vse8_v_i8m1(a_raw, a, vl);
  __riscv_vse8_v_i8m1(b_raw, b, vl);
  for (int i = 0; i < 8; ++i) {
    temp0[2 * i] = a_raw[i];
    temp0[2 * i+1] = b_raw[i];
    temp1[2 * i] = a_raw[i+8];
    temp1[2 * i+1] = b_raw[i+8];
  }
  vint8m1_t z0 = __riscv_vle8_v_i8m1(temp0, vl);
  vint8m1_t z1 = __riscv_vle8_v_i8m1(temp1, vl);*/
  //size_t vl = __riscv_vsetvl_e8m1(16);
  //vint8m1_t z0 = __riscv_vmv_v_x_i8m1(0, vl);
  //vint8m1_t z1 = __riscv_vmv_v_x_i8m1(0, vl);
  vint16m1_t r0 = __riscv_vreinterpret_v_i8m1_i16m1(z0);
  vint16m1_t r1 = __riscv_vreinterpret_v_i8m1_i16m1(z1);
  vint16m1x2_t r = __riscv_vcreate_v_i16m1x2(r0, r1);
  return r;
}

static inline vint32m1x2_t iree_uk_zip_8xi16_as_4xi32(vint16m1_t a,
                                                          vint16m1_t b) {
  /*int16x8x2_t z = vzipq_s16(a, b);
  int32x4x2_t r;
  r.val[0] = vreinterpretq_s32_s16(z.val[0]);
  r.val[1] = vreinterpretq_s32_s16(z.val[1]);*/
  size_t vl = __riscv_vsetvl_e16m1(8);
  size_t half_vl = __riscv_vsetvl_e16m1(4);
  vint16m1_t a_lo = __riscv_vslidedown_vx_i16m1(a, 0, half_vl);
  vint16m1_t a_hi = __riscv_vslidedown_vx_i16m1(a, half_vl, half_vl);
  vint16m1_t b_lo = __riscv_vslidedown_vx_i16m1(b, 0, half_vl);
  vint16m1_t b_hi = __riscv_vslidedown_vx_i16m1(b, half_vl, half_vl);
  vint16m1_t in0 = __riscv_vslideup_vx_i16m1(a_lo, b_lo, half_vl, vl);
  vint16m1_t in1 = __riscv_vslideup_vx_i16m1(a_hi, b_hi, half_vl, vl);
  uint16_t lane[8] = {0,4,1,5,2,6,3,7};
  vuint16m1_t idx = __riscv_vle16_v_u16m1(lane, vl);
  vint16m1_t z0 = __riscv_vrgather_vv_i16m1(in0, idx, vl);
  vint16m1_t z1 = __riscv_vrgather_vv_i16m1(in1, idx, vl);
  //size_t vl = __riscv_vsetvl_e16m1(8);
  //vint16m1_t z0 = __riscv_vmv_v_x_i16m1(0, vl);
  //vint16m1_t z1 = __riscv_vmv_v_x_i16m1(0, vl);
  vint32m1_t r0 = __riscv_vreinterpret_v_i16m1_i32m1(z0);
  vint32m1_t r1 = __riscv_vreinterpret_v_i16m1_i32m1(z1);
  vint32m1x2_t r = __riscv_vcreate_v_i32m1x2(r0, r1);
  return r;
}

static inline vint64m1x2_t iree_uk_zip_4xi32_as_2xi64(vint32m1_t a,
                                                          vint32m1_t b) {
  /*int32x4x2_t z = vzipq_s32(a, b);
  int64x2x2_t r;
  r.val[0] = vreinterpretq_s64_s32(z.val[0]);
  r.val[1] = vreinterpretq_s64_s32(z.val[1]);*/
  size_t vl = __riscv_vsetvl_e32m1(4);
  size_t half_vl = __riscv_vsetvl_e32m1(2);
  vint32m1_t a_lo = __riscv_vslidedown_vx_i32m1(a, 0, half_vl);
  vint32m1_t a_hi = __riscv_vslidedown_vx_i32m1(a, half_vl, half_vl);
  vint32m1_t b_lo = __riscv_vslidedown_vx_i32m1(b, 0, half_vl);
  vint32m1_t b_hi = __riscv_vslidedown_vx_i32m1(b, half_vl, half_vl);
  vint32m1_t in0 = __riscv_vslideup_vx_i32m1(a_lo, b_lo, half_vl, vl);
  vint32m1_t in1 = __riscv_vslideup_vx_i32m1(a_hi, b_hi, half_vl, vl);
  uint32_t lane[4] = {0,2,1,3};
  vuint32m1_t idx = __riscv_vle32_v_u32m1(lane, vl);
  vint32m1_t z0 = __riscv_vrgather_vv_i32m1(in0, idx, vl);
  vint32m1_t z1 = __riscv_vrgather_vv_i32m1(in1, idx, vl);
  //size_t vl = __riscv_vsetvl_e32m1(4);
  //vint32m1_t z0 = __riscv_vmv_v_x_i32m1(0, vl);
  //vint32m1_t z1 = __riscv_vmv_v_x_i32m1(0, vl);
  vint64m1_t r0 = __riscv_vreinterpret_v_i32m1_i64m1(z0);
  vint64m1_t r1 = __riscv_vreinterpret_v_i32m1_i64m1(z1);
  vint64m1x2_t r = __riscv_vcreate_v_i64m1x2(r0, r1);
  return r;
}

//[8x1_x8]
static inline void iree_uk_copy_8x1xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  /*int8x8_t v = vdup_n_s8(0);
  v = vld1_lane_s8(in_ptr + 0 * in_stride, v, 0);
  v = vld1_lane_s8(in_ptr + 1 * in_stride, v, 1);
  v = vld1_lane_s8(in_ptr + 2 * in_stride, v, 2);
  v = vld1_lane_s8(in_ptr + 3 * in_stride, v, 3);
  v = vld1_lane_s8(in_ptr + 4 * in_stride, v, 4);
  v = vld1_lane_s8(in_ptr + 5 * in_stride, v, 5);
  v = vld1_lane_s8(in_ptr + 6 * in_stride, v, 6);
  v = vld1_lane_s8(in_ptr + 7 * in_stride, v, 7);
  vst1_s8(out_ptr, v);*/
  size_t vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t lane0 = __riscv_vle8_v_i8m1(in_ptr + 0 * in_stride, vl);
  vint8m1_t lane1 = __riscv_vle8_v_i8m1(in_ptr + 1 * in_stride, vl);
  vint8m1_t lane2 = __riscv_vle8_v_i8m1(in_ptr + 2 * in_stride, vl);
  vint8m1_t lane3 = __riscv_vle8_v_i8m1(in_ptr + 3 * in_stride, vl);
  vint8m1_t lane4 = __riscv_vle8_v_i8m1(in_ptr + 4 * in_stride, vl);
  vint8m1_t lane5 = __riscv_vle8_v_i8m1(in_ptr + 5 * in_stride, vl);
  vint8m1_t lane6 = __riscv_vle8_v_i8m1(in_ptr + 6 * in_stride, vl);
  vint8m1_t lane7 = __riscv_vle8_v_i8m1(in_ptr + 7 * in_stride, vl);
  int8_t e0 = __riscv_vmv_x_s_i8m1_i8(lane0);
  int8_t e1 = __riscv_vmv_x_s_i8m1_i8(lane1);
  int8_t e2 = __riscv_vmv_x_s_i8m1_i8(lane2);
  int8_t e3 = __riscv_vmv_x_s_i8m1_i8(lane3);
  int8_t e4 = __riscv_vmv_x_s_i8m1_i8(lane4);
  int8_t e5 = __riscv_vmv_x_s_i8m1_i8(lane5);
  int8_t e6 = __riscv_vmv_x_s_i8m1_i8(lane6);
  int8_t e7 = __riscv_vmv_x_s_i8m1_i8(lane7);
  vint8m1_t temp = __riscv_vmv_v_x_i8m1(e0, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e1, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e2, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e3, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e4, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e5, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e6, vl);
  temp = __riscv_vslide1down_vx_i8m1(temp, e7, vl);
  __riscv_vse8_v_i8m1(out_ptr, temp, vl);
  /*int8_t a0[8], a1[8], a2[8], a3[8], a4[8], a5[8], a6[8], a7[8];
  __riscv_vse8_v_i8m1(a0, lane0, vl);
  __riscv_vse8_v_i8m1(a1, lane1, vl);
  __riscv_vse8_v_i8m1(a2, lane2, vl);
  __riscv_vse8_v_i8m1(a3, lane3, vl);
  __riscv_vse8_v_i8m1(a4, lane4, vl);
  __riscv_vse8_v_i8m1(a5, lane5, vl);
  __riscv_vse8_v_i8m1(a6, lane6, vl);
  __riscv_vse8_v_i8m1(a7, lane7, vl);
  int8_t temp0[8];
  temp0[0] = a0[0];
  temp0[1] = a1[0];
  temp0[2] = a2[0];
  temp0[3] = a3[0];
  temp0[4] = a4[0];
  temp0[5] = a5[0];
  temp0[6] = a6[0];
  temp0[7] = a7[0];
  vint8m1_t v = __riscv_vmv_v_x_i8m1(0, vl);
  v = __riscv_vle8_v_i8m1(temp0, vl);
  __riscv_vse8_v_i8m1(out_ptr, v, vl);*/
}

//[8x4_x8]
static inline void iree_uk_copy_8x4xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  /*int8x16x2_t in = iree_uk_load_8x4xi8_strided(in_ptr, in_stride);
  vst1q_s8(out_ptr + 0, in.val[0]);
  vst1q_s8(out_ptr + 16, in.val[1]);*/
  vint8m1x2_t in = iree_uk_load_8x4xi8_strided(in_ptr, in_stride);
  size_t vl = __riscv_vsetvl_e8m1(16);
  vint8m1_t in0 = __riscv_vget_v_i8m1x2_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x2_i8m1(in, 1);
  __riscv_vse8_v_i8m1(out_ptr + 0, in0, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16, in1, vl);
}

//[8x8_x8]
static inline void iree_uk_copy_8x8xi8_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  /*int8x16x4_t in = iree_uk_load_8x8xi8_strided(in_ptr, in_stride);
  vst1q_s8(out_ptr + 0, in.val[0]);
  vst1q_s8(out_ptr + 16, in.val[1]);
  vst1q_s8(out_ptr + 32, in.val[2]);
  vst1q_s8(out_ptr + 48, in.val[3]);*/
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided(in_ptr, in_stride);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  size_t vl = __riscv_vsetvl_e8m1(16);
  __riscv_vse8_v_i8m1(out_ptr + 0, in0, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16, in1, vl);
  __riscv_vse8_v_i8m1(out_ptr + 32, in2, vl);
  __riscv_vse8_v_i8m1(out_ptr + 48, in3, vl);
}

//[8x4_x8]
static inline void
iree_uk_copy_8x8xi8_tiled_1x4_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  /*int8x16x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 2, 1, 3, 4, 6, 5, 7);
  int32x4x2_t c0 = vtrnq_s32(vreinterpretq_s32_s8(in.val[0]),
                             vreinterpretq_s32_s8(in.val[1]));
  int32x4x2_t c1 = vtrnq_s32(vreinterpretq_s32_s8(in.val[2]),
                             vreinterpretq_s32_s8(in.val[3]));
  vst1q_s8(out_ptr + 0 + 0 * out_stride, vreinterpretq_s8_s32(c0.val[0]));
  vst1q_s8(out_ptr + 16 + 0 * out_stride, vreinterpretq_s8_s32(c1.val[0]));
  vst1q_s8(out_ptr + 0 + 1 * out_stride, vreinterpretq_s8_s32(c0.val[1]));
  vst1q_s8(out_ptr + 16 + 1 * out_stride, vreinterpretq_s8_s32(c1.val[1]));*/
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 2, 1, 3, 4, 6, 5, 7);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  vint32m1_t row0 = __riscv_vreinterpret_v_i8m1_i32m1(in0);
  vint32m1_t row1 = __riscv_vreinterpret_v_i8m1_i32m1(in1);
  vint32m1_t row2 = __riscv_vreinterpret_v_i8m1_i32m1(in2);
  vint32m1_t row3 = __riscv_vreinterpret_v_i8m1_i32m1(in3);
  size_t vl = __riscv_vsetvl_e32m1(8);
  size_t half_vl = __riscv_vsetvl_e32m1(4);
  vint32m1_t temp0 = __riscv_vslideup_vx_i32m1(row0, row1, half_vl, vl);
  vint32m1_t temp1 = __riscv_vslideup_vx_i32m1(row2, row3, half_vl, vl);
  vl = __riscv_vsetvl_e32m1(4);
  uint32_t lane0[4] = {0,4,2,6};
  vuint32m1_t idx0 = __riscv_vle32_v_u32m1(lane0, vl);
  uint32_t lane1[4] = {1,5,3,7};
  vuint32m1_t idx1 = __riscv_vle32_v_u32m1(lane1, vl);
  vint32m1_t c0_0 = __riscv_vrgather_vv_i32m1(temp0, idx0, vl);
  vint32m1_t c0_1 = __riscv_vrgather_vv_i32m1(temp0, idx1, vl);
  vint32m1_t c1_0 = __riscv_vrgather_vv_i32m1(temp1, idx0, vl);
  vint32m1_t c1_1 = __riscv_vrgather_vv_i32m1(temp1, idx1, vl);
  vint8m1_t c0_0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c0_0);
  vint8m1_t c1_0_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c1_0);
  vint8m1_t c0_1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c0_1);
  vint8m1_t c1_1_i8 = __riscv_vreinterpret_v_i32m1_i8m1(c1_1);
  vl = __riscv_vsetvl_e8m1(16);
  __riscv_vse8_v_i8m1(out_ptr + 0 + 0 * out_stride, c0_0_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16 + 0 * out_stride, c1_0_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 0 + 1 * out_stride, c0_1_i8, vl);
  __riscv_vse8_v_i8m1(out_ptr + 16 + 1 * out_stride, c1_1_i8, vl);
  //c0_0 = [a0_0, a1_0, a0_2, a1_2]
  //c0_1 = [a0_1, a1_1, a0_3, a1_3]
  //c1_0 = [a2_0, a3_0, a2_2, a3_2]
  //c1_1 = [a2_1, a3_1, a2_3, a3_3]
  /*int32_t a0[4], a1[4], a2[4], a3[4];
  __riscv_vse32_v_i32m1(a0, in0_i32, vl);
  __riscv_vse32_v_i32m1(a1, in1_i32, vl);
  __riscv_vse32_v_i32m1(a2, in2_i32, vl);
  __riscv_vse32_v_i32m1(a3, in3_i32, vl);
  int32_t c0_0[4], c0_1[4], c1_0[4], c1_1[4];
  for (int i = 0; i < 2; ++i) {
    c0_0[2 * i] = a0[2 * i];
    c0_0[2 * i + 1] = a1[2 * i];
    c0_1[2 * i] = a0[2 * i + 1];
    c0_1[2 * i + 1] = a1[2 * i + 1];
    c1_0[2 * i] = a2[2 * i];
    c1_0[2 * i + 1] = a3[2 * i];
    c1_1[2 * i] = a2[2 * i + 1];
    c1_1[2 * i + 1] = a3[2 * i + 1];
  }
  vint32m1_t in0_0 = __riscv_vle32_v_i32m1(c0_0, vl);
  vint32m1_t in1_0 = __riscv_vle32_v_i32m1(c1_0, vl);
  vint32m1_t in0_1 = __riscv_vle32_v_i32m1(c0_1, vl);
  vint32m1_t in1_1 = __riscv_vle32_v_i32m1(c1_1, vl);*/
}

static inline void iree_uk_copy_8x32xi8_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  for (int i = 0; i < 8; ++i) {
    //iree_uk_memcpy(out_ptr + i * out_stride, in_ptr + i * in_stride, 32);
    size_t vl = __riscv_vsetvl_e8m4(32);
    vint8m4_t vec = __riscv_vle8_v_i8m4(in_ptr + i * in_stride, vl);
    __riscv_vse8_v_i8m4(out_ptr + i * out_stride, vec, vl);
  }
}

//[8x1_x8 8x8_x8]
static inline void iree_uk_copy_8x8xi8_transpose_strided_to_strided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t out_stride,
    iree_uk_index_t in_stride) {
  /*int8x16x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 4, 1, 5, 2, 6, 3, 7);
  int16x8x2_t zip_i16_0 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[0], in.val[1]);
  int16x8x2_t zip_i16_1 = iree_uk_neon_zip_16xi8_as_8xi16(in.val[2], in.val[3]);
  int32x4x2_t zip_i32_0 =
      iree_uk_neon_zip_8xi16_as_4xi32(zip_i16_0.val[0], zip_i16_1.val[0]);
  int32x4x2_t zip_i32_1 =
      iree_uk_neon_zip_8xi16_as_4xi32(zip_i16_0.val[1], zip_i16_1.val[1]);
  int64x2x2_t zip_i64_0 =
      iree_uk_neon_zip_4xi32_as_2xi64(zip_i32_0.val[0], zip_i32_1.val[0]);
  int64x2x2_t zip_i64_1 =
      iree_uk_neon_zip_4xi32_as_2xi64(zip_i32_0.val[1], zip_i32_1.val[1]);
  int8x16x4_t out;
  out.val[0] = vreinterpretq_s8_s64(zip_i64_0.val[0]);
  out.val[1] = vreinterpretq_s8_s64(zip_i64_0.val[1]);
  out.val[2] = vreinterpretq_s8_s64(zip_i64_1.val[0]);
  out.val[3] = vreinterpretq_s8_s64(zip_i64_1.val[1]);
  vst1_s8(out_ptr + 0 * out_stride, vget_low_s8(out.val[0]));
  vst1_s8(out_ptr + 1 * out_stride, vget_high_s8(out.val[0]));
  vst1_s8(out_ptr + 2 * out_stride, vget_low_s8(out.val[1]));
  vst1_s8(out_ptr + 3 * out_stride, vget_high_s8(out.val[1]));
  vst1_s8(out_ptr + 4 * out_stride, vget_low_s8(out.val[2]));
  vst1_s8(out_ptr + 5 * out_stride, vget_high_s8(out.val[2]));
  vst1_s8(out_ptr + 6 * out_stride, vget_low_s8(out.val[3]));
  vst1_s8(out_ptr + 7 * out_stride, vget_high_s8(out.val[3]));*/
  vint8m1x4_t in = iree_uk_load_8x8xi8_strided_permute(
      in_ptr, in_stride, 0, 4, 1, 5, 2, 6, 3, 7);
  vint8m1_t in0 = __riscv_vget_v_i8m1x4_i8m1(in, 0);
  vint8m1_t in1 = __riscv_vget_v_i8m1x4_i8m1(in, 1);
  vint8m1_t in2 = __riscv_vget_v_i8m1x4_i8m1(in, 2);
  vint8m1_t in3 = __riscv_vget_v_i8m1x4_i8m1(in, 3);
  vint16m1x2_t zip_i16_0 = iree_uk_zip_16xi8_as_8xi16(in0, in1);
  vint16m1x2_t zip_i16_1 = iree_uk_zip_16xi8_as_8xi16(in2, in3);
  vint16m1_t zip_i16_00 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_0, 0);
  vint16m1_t zip_i16_01 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_0, 1);
  vint16m1_t zip_i16_10 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_1, 0);
  vint16m1_t zip_i16_11 = __riscv_vget_v_i16m1x2_i16m1(zip_i16_1, 1);
  vint32m1x2_t zip_i32_0 =
      iree_uk_zip_8xi16_as_4xi32(zip_i16_00, zip_i16_10);
  vint32m1x2_t zip_i32_1 =
      iree_uk_zip_8xi16_as_4xi32(zip_i16_01, zip_i16_11);
  vint32m1_t zip_i32_00 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_0, 0);
  vint32m1_t zip_i32_01 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_0, 1);
  vint32m1_t zip_i32_10 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_1, 0);
  vint32m1_t zip_i32_11 = __riscv_vget_v_i32m1x2_i32m1(zip_i32_1, 1);
  vint64m1x2_t zip_i64_0 =
      iree_uk_zip_4xi32_as_2xi64(zip_i32_00, zip_i32_10);
  vint64m1x2_t zip_i64_1 =
      iree_uk_zip_4xi32_as_2xi64(zip_i32_01, zip_i32_11);
  vint64m1_t zip_i64_00 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_0, 0);
  vint64m1_t zip_i64_01 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_0, 1);
  vint64m1_t zip_i64_10 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_1, 0);
  vint64m1_t zip_i64_11 = __riscv_vget_v_i64m1x2_i64m1(zip_i64_1, 1);
  vint8m1_t out0 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_00);
  vint8m1_t out1 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_01);
  vint8m1_t out2 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_10);
  vint8m1_t out3 = __riscv_vreinterpret_v_i64m1_i8m1(zip_i64_11);
  size_t vl = __riscv_vsetvl_e8m1(16);
  size_t half_vl = __riscv_vsetvl_e8m1(8);
  vint8m1_t out0_lo = __riscv_vslidedown_vx_i8m1(out0, 0, half_vl);
  vint8m1_t out0_hi = __riscv_vslidedown_vx_i8m1(out0, half_vl, half_vl);
  vint8m1_t out1_lo = __riscv_vslidedown_vx_i8m1(out1, 0, half_vl);
  vint8m1_t out1_hi = __riscv_vslidedown_vx_i8m1(out1, half_vl, half_vl);
  vint8m1_t out2_lo = __riscv_vslidedown_vx_i8m1(out2, 0, half_vl);
  vint8m1_t out2_hi = __riscv_vslidedown_vx_i8m1(out2, half_vl, half_vl);
  vint8m1_t out3_lo = __riscv_vslidedown_vx_i8m1(out3, 0, half_vl);
  vint8m1_t out3_hi = __riscv_vslidedown_vx_i8m1(out3, half_vl, half_vl);
  vl = __riscv_vsetvl_e8m1(8);
  __riscv_vse8_v_i8m1(out_ptr + 0 * out_stride, out0_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 1 * out_stride, out0_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 2 * out_stride, out1_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 3 * out_stride, out1_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 4 * out_stride, out2_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 5 * out_stride, out2_hi, vl);
  __riscv_vse8_v_i8m1(out_ptr + 6 * out_stride, out3_lo, vl);
  __riscv_vse8_v_i8m1(out_ptr + 7 * out_stride, out3_hi, vl);
}

static inline void iree_uk_copy_8x8xi8_transpose_strided_to_unstrided(
    iree_uk_int8_t* IREE_UK_RESTRICT out_ptr,
    const iree_uk_int8_t* IREE_UK_RESTRICT in_ptr, iree_uk_index_t in_stride) {
  // Clang (Android NDK r25) actually produces worse code when this code is
  // specialized for out_stride==8 using longer contiguous stores!
  iree_uk_copy_8x8xi8_transpose_strided_to_strided(out_ptr, in_ptr, 8,
                                                        in_stride);
}

#endif  // IREE_BUILTINS_UKERNEL_ARCH_RISCV_64_COMMON_RISCV_64_H_
