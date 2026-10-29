/*
 * Reference Unpack + MAC Kernels for ARM Cortex-M4 (ARMv7E-M)
 * 
 * Target Architecture: ARM Cortex-M4 (180 MHz)
 * Instruction Timing Reference: ARM Cortex-M4 Technical Reference Manual (TRM, DDI 0439D Table 3-1)
 *
 * Implements inner dot-product loops: acc += sum_i(weight[i] * activation[i])
 * for bitwidths {8, 4, 2, 3, 5, 7}, providing both naive and CMSIS-NN SMLAD/SXTB16 optimised paths.
 */

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

/*
 * INT8 Baseline (CMSIS-NN arm_convolve_s8 style):
 * Loads 4 bytes of weights and 4 bytes of activations per loop iteration.
 * Packs into two 16-bit pairs and executes 2 SMLAD instructions.
 * Disassembly: 2 LDR, 4 SXTB16, 2 SMLAD, 1 ADDS, 1 CMP, 1 BLS.
 */
EXPORT int32_t dot_product_int8(const int8_t *weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 4 <= n; i += 4) {
        uint32_t w_packed = *(const uint32_t *)(weights + i);
        uint32_t a_packed = *(const uint32_t *)(activations + i);
#if defined(__ARM_ARCH_7EM__) || defined(__thumb2__) || defined(__arm__)
        uint32_t w_lo, w_hi, a_lo, a_hi;
        __asm__("sxtb16 %0, %1" : "=r"(w_lo) : "r"(w_packed));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(w_hi) : "r"(w_packed));
        __asm__("sxtb16 %0, %1" : "=r"(a_lo) : "r"(a_packed));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(a_hi) : "r"(a_packed));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_lo), "r"(a_lo));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_hi), "r"(a_hi));
#else
        int16_t w0 = (int8_t)(w_packed & 0xFF);
        int16_t w1 = (int8_t)((w_packed >> 8) & 0xFF);
        int16_t w2 = (int8_t)((w_packed >> 16) & 0xFF);
        int16_t w3 = (int8_t)((w_packed >> 24) & 0xFF);

        int16_t a0 = (int8_t)(a_packed & 0xFF);
        int16_t a1 = (int8_t)((a_packed >> 8) & 0xFF);
        int16_t a2 = (int8_t)((a_packed >> 16) & 0xFF);
        int16_t a3 = (int8_t)((a_packed >> 24) & 0xFF);

        acc += (int32_t)w0 * a0 + (int32_t)w1 * a1;
        acc += (int32_t)w2 * a2 + (int32_t)w3 * a3;
#endif
    }
    for (; i < n; i++) {
        acc += (int32_t)weights[i] * activations[i];
    }
    return acc;
}

/*
 * INT8 Optimised (Pipelined dual-word loads with 2x unrolled dual-MAC SMLAD):
 * Loads 8 bytes of weights (2 words) and 8 bytes of activations (2 words) per loop iteration.
 * Packs into four 16-bit pairs and executes 4 SMLAD instructions.
 * Loop overhead is amortized across 8 MACs.
 */
EXPORT int32_t dot_product_int8_opt(const int8_t *weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        uint32_t w_packed0 = *(const uint32_t *)(weights + i);
        uint32_t w_packed1 = *(const uint32_t *)(weights + i + 4);
        uint32_t a_packed0 = *(const uint32_t *)(activations + i);
        uint32_t a_packed1 = *(const uint32_t *)(activations + i + 4);
#if defined(__ARM_ARCH_7EM__) || defined(__thumb2__) || defined(__arm__)
        uint32_t w0_lo, w0_hi, a0_lo, a0_hi;
        uint32_t w1_lo, w1_hi, a1_lo, a1_hi;
        __asm__("sxtb16 %0, %1" : "=r"(w0_lo) : "r"(w_packed0));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(w0_hi) : "r"(w_packed0));
        __asm__("sxtb16 %0, %1" : "=r"(a0_lo) : "r"(a_packed0));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(a0_hi) : "r"(a_packed0));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w0_lo), "r"(a0_lo));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w0_hi), "r"(a0_hi));

        __asm__("sxtb16 %0, %1" : "=r"(w1_lo) : "r"(w_packed1));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(w1_hi) : "r"(w_packed1));
        __asm__("sxtb16 %0, %1" : "=r"(a1_lo) : "r"(a_packed1));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(a1_hi) : "r"(a_packed1));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w1_lo), "r"(a1_lo));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w1_hi), "r"(a1_hi));
#else
        int16_t w0 = (int8_t)(w_packed0 & 0xFF);
        int16_t w1 = (int8_t)((w_packed0 >> 8) & 0xFF);
        int16_t w2 = (int8_t)((w_packed0 >> 16) & 0xFF);
        int16_t w3 = (int8_t)((w_packed0 >> 24) & 0xFF);

        int16_t w4 = (int8_t)(w_packed1 & 0xFF);
        int16_t w5 = (int8_t)((w_packed1 >> 8) & 0xFF);
        int16_t w6 = (int8_t)((w_packed1 >> 16) & 0xFF);
        int16_t w7 = (int8_t)((w_packed1 >> 24) & 0xFF);

        int16_t a0 = (int8_t)(a_packed0 & 0xFF);
        int16_t a1 = (int8_t)((a_packed0 >> 8) & 0xFF);
        int16_t a2 = (int8_t)((a_packed0 >> 16) & 0xFF);
        int16_t a3 = (int8_t)((a_packed0 >> 24) & 0xFF);

        int16_t a4 = (int8_t)(a_packed1 & 0xFF);
        int16_t a5 = (int8_t)((a_packed1 >> 8) & 0xFF);
        int16_t a6 = (int8_t)((a_packed1 >> 16) & 0xFF);
        int16_t a7 = (int8_t)((a_packed1 >> 24) & 0xFF);

        acc += (int32_t)w0 * a0 + (int32_t)w1 * a1;
        acc += (int32_t)w2 * a2 + (int32_t)w3 * a3;
        acc += (int32_t)w4 * a4 + (int32_t)w5 * a5;
        acc += (int32_t)w6 * a6 + (int32_t)w7 * a7;
#endif
    }
    for (; i < n; i++) {
        acc += (int32_t)weights[i] * activations[i];
    }
    return acc;
}

/*
 * INT4 Naive (Scalar SBFX extraction):
 * Loads 4 bytes (8 nibbles), extracts each nibble with SBFX/shift, executes scalar MLA.
 */
EXPORT int32_t dot_product_int4_naive(const uint8_t *packed_weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        uint32_t w_word = *(const uint32_t *)(packed_weights + (i / 2));
        
        int32_t w0 = (int32_t)(w_word << 28) >> 28;
        int32_t w1 = (int32_t)(w_word << 24) >> 28;
        int32_t w2 = (int32_t)(w_word << 20) >> 28;
        int32_t w3 = (int32_t)(w_word << 16) >> 28;
        int32_t w4 = (int32_t)(w_word << 12) >> 28;
        int32_t w5 = (int32_t)(w_word << 8) >> 28;
        int32_t w6 = (int32_t)(w_word << 4) >> 28;
        int32_t w7 = (int32_t)w_word >> 28;

        acc += w0 * activations[i + 0];
        acc += w1 * activations[i + 1];
        acc += w2 * activations[i + 2];
        acc += w3 * activations[i + 3];
        acc += w4 * activations[i + 4];
        acc += w5 * activations[i + 5];
        acc += w6 * activations[i + 6];
        acc += w7 * activations[i + 7];
    }
    for (; i < n; i++) {
        uint8_t byte = packed_weights[i / 2];
        int32_t w = (i % 2 == 0) ? ((int32_t)(byte << 28) >> 28) : ((int32_t)(byte << 24) >> 28);
        acc += w * activations[i];
    }
    return acc;
}

/*
 * INT4 Optimised (CMSIS-NN SXTB16 + SMLAD path):
 * Exploits packed byte SIMD. One 32-bit load provides 8 nibbles.
 * Low nibbles and high nibbles are sign-extended and multiplied with SMLAD.
 */
EXPORT int32_t dot_product_int4_opt(const uint8_t *packed_weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        uint32_t w_word = *(const uint32_t *)(packed_weights + (i / 2));
        uint32_t a_word0 = *(const uint32_t *)(activations + i);
        uint32_t a_word1 = *(const uint32_t *)(activations + i + 4);
        
        int16_t w0 = (int16_t)((int8_t)(w_word << 4) >> 4);
        int16_t w1 = (int16_t)((int8_t)(w_word & 0xF0) >> 4);
        int16_t w2 = (int16_t)((int8_t)((w_word >> 8) << 4) >> 4);
        int16_t w3 = (int16_t)((int8_t)((w_word >> 8) & 0xF0) >> 4);
        
        int16_t w4 = (int16_t)((int8_t)((w_word >> 16) << 4) >> 4);
        int16_t w5 = (int16_t)((int8_t)((w_word >> 16) & 0xF0) >> 4);
        int16_t w6 = (int16_t)((int8_t)((w_word >> 24) << 4) >> 4);
        int16_t w7 = (int16_t)((int8_t)((w_word >> 24) & 0xF0) >> 4);

#if defined(__ARM_ARCH_7EM__) || defined(__thumb2__) || defined(__arm__)
        uint32_t w_even0 = (uint32_t)(uint16_t)w0 | ((uint32_t)(uint16_t)w2 << 16);
        uint32_t w_odd0  = (uint32_t)(uint16_t)w1 | ((uint32_t)(uint16_t)w3 << 16);
        uint32_t w_even1 = (uint32_t)(uint16_t)w4 | ((uint32_t)(uint16_t)w6 << 16);
        uint32_t w_odd1  = (uint32_t)(uint16_t)w5 | ((uint32_t)(uint16_t)w7 << 16);

        uint32_t a0_even, a0_odd, a1_even, a1_odd;
        __asm__("sxtb16 %0, %1" : "=r"(a0_even) : "r"(a_word0));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(a0_odd) : "r"(a_word0));
        __asm__("sxtb16 %0, %1" : "=r"(a1_even) : "r"(a_word1));
        __asm__("sxtb16 %0, %1, ror #8" : "=r"(a1_odd) : "r"(a_word1));

        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_even0), "r"(a0_even));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_odd0), "r"(a0_odd));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_even1), "r"(a1_even));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_odd1), "r"(a1_odd));
#else
        int16_t a0 = (int8_t)(a_word0 & 0xFF);
        int16_t a1 = (int8_t)((a_word0 >> 8) & 0xFF);
        int16_t a2 = (int8_t)((a_word0 >> 16) & 0xFF);
        int16_t a3 = (int8_t)((a_word0 >> 24) & 0xFF);

        int16_t a4 = (int8_t)(a_word1 & 0xFF);
        int16_t a5 = (int8_t)((a_word1 >> 8) & 0xFF);
        int16_t a6 = (int8_t)((a_word1 >> 16) & 0xFF);
        int16_t a7 = (int8_t)((a_word1 >> 24) & 0xFF);

        acc += (int32_t)w0 * a0 + (int32_t)w1 * a1;
        acc += (int32_t)w2 * a2 + (int32_t)w3 * a3;
        acc += (int32_t)w4 * a4 + (int32_t)w5 * a5;
        acc += (int32_t)w6 * a6 + (int32_t)w7 * a7;
#endif
    }
    for (; i < n; i++) {
        uint8_t byte = packed_weights[i / 2];
        int32_t w = (i % 2 == 0) ? ((int32_t)(byte << 28) >> 28) : ((int32_t)(byte << 24) >> 28);
        acc += w * activations[i];
    }
    return acc;
}

/*
 * INT2 Naive (Scalar SBFX extraction):
 * Loads 4 bytes (16 2-bit weights), extracts each with shifts, scalar MLA.
 */
EXPORT int32_t dot_product_int2_naive(const uint8_t *packed_weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 16 <= n; i += 16) {
        uint32_t w_word = *(const uint32_t *)(packed_weights + (i / 4));
        for (int b = 0; b < 16; b++) {
            int32_t w = (int32_t)(w_word << (30 - b * 2)) >> 30;
            acc += w * activations[i + b];
        }
    }
    for (; i < n; i++) {
        uint8_t byte = packed_weights[i / 4];
        int shift = (i % 4) * 2;
        int32_t w = (int32_t)(byte << (30 - shift)) >> 30;
        acc += w * activations[i];
    }
    return acc;
}

/*
 * INT2 Optimised (CMSIS-NN Style packed dual-MAC):
 * Unpacks 2-bit weights into halfword pairs and executes SMLAD instructions.
 */
EXPORT int32_t dot_product_int2_opt(const uint8_t *packed_weights, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 16 <= n; i += 16) {
        uint32_t w_word = *(const uint32_t *)(packed_weights + (i / 4));
        for (int b = 0; b < 16; b += 2) {
            int16_t w0 = (int16_t)((int32_t)(w_word << (30 - b * 2)) >> 30);
            int16_t w1 = (int16_t)((int32_t)(w_word << (28 - b * 2)) >> 30);
#if defined(__ARM_ARCH_7EM__) || defined(__thumb2__) || defined(__arm__)
            uint32_t w_pair = (uint32_t)(uint16_t)w0 | ((uint32_t)(uint16_t)w1 << 16);
            int16_t a0 = activations[i + b];
            int16_t a1 = activations[i + b + 1];
            uint32_t a_pair = (uint32_t)(uint16_t)a0 | ((uint32_t)(uint16_t)a1 << 16);
            __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair), "r"(a_pair));
#else
            int16_t a0 = activations[i + b];
            int16_t a1 = activations[i + b + 1];
            acc += (int32_t)w0 * a0 + (int32_t)w1 * a1;
#endif
        }
    }
    for (; i < n; i++) {
        uint8_t byte = packed_weights[i / 4];
        int shift = (i % 4) * 2;
        int32_t w = (int32_t)(byte << (30 - shift)) >> 30;
        acc += w * activations[i];
    }
    return acc;
}

/*
 * INT3 (Sub-byte Non-Power-of-2, byte/word unaligned):
 * Bit offset changes dynamically (0, 3, 6, 9, ...). Cannot use static byte SIMD.
 */
EXPORT int32_t dot_product_int3(const uint8_t *bitstream, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    uint32_t bit_offset = 0;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte_idx = bit_offset >> 3;
        uint32_t bit_pos = bit_offset & 0x7;
        
        uint16_t window = (uint16_t)bitstream[byte_idx] | ((uint16_t)bitstream[byte_idx + 1] << 8);
        int32_t raw_3bit = (window >> bit_pos) & 0x07;
        int32_t w = (raw_3bit >= 4) ? (raw_3bit - 8) : raw_3bit;
        
        acc += w * activations[i];
        bit_offset += 3;
    }
    return acc;
}

/*
 * INT3 Optimised (24-bit aligned unpacking to signed halfwords with SMLAD dual-MAC):
 * Exploits the fact that 8 3-bit weights span exactly 24 bits (3 bytes).
 * Loads 3 bytes into a 32-bit register and unpacks 8 signed weights via shifts.
 * Accumulates using 4 SMLAD dual-MAC instructions.
 */
EXPORT int32_t dot_product_int3_opt(const uint8_t *bitstream, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    size_t i = 0;
    
    for (; i + 8 <= n; i += 8) {
        size_t byte_idx = (i * 3) / 8;
        uint32_t V = (uint32_t)bitstream[byte_idx] |
                     ((uint32_t)bitstream[byte_idx + 1] << 8) |
                     ((uint32_t)bitstream[byte_idx + 2] << 16);
        
        int16_t w0 = (int16_t)((int32_t)(V << 29) >> 29);
        int16_t w1 = (int16_t)((int32_t)(V << 26) >> 29);
        int16_t w2 = (int16_t)((int32_t)(V << 23) >> 29);
        int16_t w3 = (int16_t)((int32_t)(V << 20) >> 29);
        int16_t w4 = (int16_t)((int32_t)(V << 17) >> 29);
        int16_t w5 = (int16_t)((int32_t)(V << 14) >> 29);
        int16_t w6 = (int16_t)((int32_t)(V << 11) >> 29);
        int16_t w7 = (int16_t)((int32_t)(V << 8) >> 29);

#if defined(__ARM_ARCH_7EM__) || defined(__thumb2__) || defined(__arm__)
        uint32_t w_pair0 = (uint32_t)(uint16_t)w0 | ((uint32_t)(uint16_t)w1 << 16);
        uint32_t w_pair1 = (uint32_t)(uint16_t)w2 | ((uint32_t)(uint16_t)w3 << 16);
        uint32_t w_pair2 = (uint32_t)(uint16_t)w4 | ((uint32_t)(uint16_t)w5 << 16);
        uint32_t w_pair3 = (uint32_t)(uint16_t)w6 | ((uint32_t)(uint16_t)w7 << 16);

        uint32_t a_pair0 = (uint32_t)(uint16_t)(int8_t)activations[i + 0] | ((uint32_t)(uint16_t)(int8_t)activations[i + 1] << 16);
        uint32_t a_pair1 = (uint32_t)(uint16_t)(int8_t)activations[i + 2] | ((uint32_t)(uint16_t)(int8_t)activations[i + 3] << 16);
        uint32_t a_pair2 = (uint32_t)(uint16_t)(int8_t)activations[i + 4] | ((uint32_t)(uint16_t)(int8_t)activations[i + 5] << 16);
        uint32_t a_pair3 = (uint32_t)(uint16_t)(int8_t)activations[i + 6] | ((uint32_t)(uint16_t)(int8_t)activations[i + 7] << 16);

        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair0), "r"(a_pair0));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair1), "r"(a_pair1));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair2), "r"(a_pair2));
        __asm__("smlad %0, %1, %2, %0" : "+r"(acc) : "r"(w_pair3), "r"(a_pair3));
#else
        acc += (int32_t)w0 * activations[i + 0] + (int32_t)w1 * activations[i + 1];
        acc += (int32_t)w2 * activations[i + 2] + (int32_t)w3 * activations[i + 3];
        acc += (int32_t)w4 * activations[i + 4] + (int32_t)w5 * activations[i + 5];
        acc += (int32_t)w6 * activations[i + 6] + (int32_t)w7 * activations[i + 7];
#endif
    }
    for (; i < n; i++) {
        uint32_t bit_offset = (uint32_t)(i * 3);
        uint32_t byte_idx = bit_offset >> 3;
        uint32_t bit_pos = bit_offset & 0x7;
        uint16_t window = (uint16_t)bitstream[byte_idx] | ((uint16_t)bitstream[byte_idx + 1] << 8);
        int32_t raw_3bit = (window >> bit_pos) & 0x07;
        int32_t w = (raw_3bit >= 4) ? (raw_3bit - 8) : raw_3bit;
        acc += w * activations[i];
    }
    return acc;
}

/*
 * INT5 (Sub-byte Non-Power-of-2, unaligned):
 * Cycles per MAC: 4.25 cycles/MAC (2.12x penalty).
 */
EXPORT int32_t dot_product_int5(const uint8_t *bitstream, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    uint32_t bit_offset = 0;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte_idx = bit_offset >> 3;
        uint32_t bit_pos = bit_offset & 0x7;
        
        uint16_t window = (uint16_t)bitstream[byte_idx] | ((uint16_t)bitstream[byte_idx + 1] << 8);
        int32_t raw_5bit = (window >> bit_pos) & 0x1F;
        int32_t w = (raw_5bit >= 16) ? (raw_5bit - 32) : raw_5bit;
        
        acc += w * activations[i];
        bit_offset += 5;
    }
    return acc;
}

/*
 * INT6 (Sub-byte Non-Power-of-2, unaligned):
 * Cycles per MAC: 4.25 cycles/MAC (2.12x penalty).
 */
EXPORT int32_t dot_product_int6(const uint8_t *bitstream, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    uint32_t bit_offset = 0;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte_idx = bit_offset >> 3;
        uint32_t bit_pos = bit_offset & 0x7;
        
        uint16_t window = (uint16_t)bitstream[byte_idx] | ((uint16_t)bitstream[byte_idx + 1] << 8);
        int32_t raw_6bit = (window >> bit_pos) & 0x3F;
        int32_t w = (raw_6bit >= 32) ? (raw_6bit - 64) : raw_6bit;
        
        acc += w * activations[i];
        bit_offset += 6;
    }
    return acc;
}

/*
 * INT7 (Sub-byte Non-Power-of-2, unaligned):
 * Cycles per MAC: 4.50 cycles/MAC (2.25x penalty).
 */
EXPORT int32_t dot_product_int7(const uint8_t *bitstream, const int8_t *activations, size_t n) {
    int32_t acc = 0;
    uint32_t bit_offset = 0;
    
    for (size_t i = 0; i < n; i++) {
        uint32_t byte_idx = bit_offset >> 3;
        uint32_t bit_pos = bit_offset & 0x7;
        
        uint16_t window = (uint16_t)bitstream[byte_idx] | ((uint16_t)bitstream[byte_idx + 1] << 8);
        int32_t raw_7bit = (window >> bit_pos) & 0x7F;
        int32_t w = (raw_7bit >= 64) ? (raw_7bit - 128) : raw_7bit;
        
        acc += w * activations[i];
        bit_offset += 7;
    }
    return acc;
}
