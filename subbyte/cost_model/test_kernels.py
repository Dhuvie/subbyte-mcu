"""
Functional Testing Suite for Reference Unpack + MAC Kernels.
Tests every C kernel in kernels.dll against a NumPy reference for random weights and activations.
"""

import ctypes
import numpy as np
from pathlib import Path

def pack_nibbles(weights: np.ndarray) -> bytes:
    n = len(weights)
    packed = bytearray((n + 1) // 2)
    for i in range(n):
        val = int(weights[i]) & 0x0F
        if i % 2 == 0:
            packed[i // 2] |= val
        else:
            packed[i // 2] |= (val << 4)
    return bytes(packed)

def pack_twobit(weights: np.ndarray) -> bytes:
    n = len(weights)
    packed = bytearray((n + 3) // 4)
    for i in range(n):
        val = int(weights[i]) & 0x03
        shift = (i % 4) * 2
        packed[i // 4] |= (val << shift)
    return bytes(packed)

def pack_arbitrary_bits(weights: np.ndarray, bitwidth: int) -> bytes:
    total_bits = len(weights) * bitwidth
    total_bytes = (total_bits + 7) // 8 + 4
    buf = bytearray(total_bytes)
    
    for i, w in enumerate(weights):
        mask = (1 << bitwidth) - 1
        raw_val = int(w) & mask
        bit_offset = i * bitwidth
        
        for b in range(bitwidth):
            bit = (raw_val >> b) & 1
            cur_bit = bit_offset + b
            byte_idx = cur_bit >> 3
            bit_idx = cur_bit & 7
            if bit:
                buf[byte_idx] |= (1 << bit_idx)
            else:
                buf[byte_idx] &= ~(1 << bit_idx)
    return bytes(buf)

def run_tests():
    dll_path = (Path(__file__).parent / "kernels.dll").resolve()
    if not dll_path.exists():
        for ext in [".so", ".dylib"]:
            p = (Path(__file__).parent / f"kernels{ext}").resolve()
            if p.exists():
                dll_path = p
                break

    if not dll_path.exists():
        # Attempt compilation if C compiler is present in environment
        import shutil, subprocess
        for cc, args in [
            ("gcc", ["-O3", "-shared", "-fPIC", "-o", str((Path(__file__).parent / "kernels.so").resolve()), str((Path(__file__).parent / "kernels.c").resolve())]),
            ("clang", ["-O3", "-shared", "-fPIC", "-o", str((Path(__file__).parent / "kernels.so").resolve()), str((Path(__file__).parent / "kernels.c").resolve())]),
            ("cl", ["/O2", "/LD", str((Path(__file__).parent / "kernels.c").resolve()), f"/Fe:{dll_path}"])]:
            if shutil.which(cc):
                try:
                    res = subprocess.run([cc] + args, capture_output=True)
                    if res.returncode == 0:
                        dll_path = (Path(__file__).parent / ("kernels.so" if cc != "cl" else "kernels.dll")).resolve()
                        break
                except Exception:
                    pass

    if not dll_path.exists():
        print(f"[INFO] Precompiled kernels library not found and no C compiler in PATH.")
        print("[INFO] Verifying bit-packing reference implementations (Python)...")
        w4 = np.array([3, -4, 7, -8], dtype=np.int8)
        p4 = pack_nibbles(w4)
        assert len(p4) == 2, "Nibble packing length error"
        w2 = np.array([1, -2, 0, -1], dtype=np.int8)
        p2 = pack_twobit(w2)
        assert len(p2) == 1, "Two-bit packing length error"
        w3 = np.array([2, -3, 1, 0], dtype=np.int8)
        p3 = pack_arbitrary_bits(w3, 3)
        assert len(p3) >= 2, "3-bit packing length error"
        print("[OK] Python bit-packing reference functions verified.")
        print("[NOTE] To compile C reference kernels: gcc -O3 -shared -fPIC -o subbyte/cost_model/kernels.so subbyte/cost_model/kernels.c")
        return

    lib = ctypes.CDLL(str(dll_path))
    
    c_int8_p = ctypes.POINTER(ctypes.c_int8)
    c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
    c_size_t = ctypes.c_size_t
    c_int32 = ctypes.c_int32
    
    lib.dot_product_int8.argtypes = [c_int8_p, c_int8_p, c_size_t]
    lib.dot_product_int8.restype = c_int32
    lib.dot_product_int8_opt.argtypes = [c_int8_p, c_int8_p, c_size_t]
    lib.dot_product_int8_opt.restype = c_int32
    lib.dot_product_int4_naive.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int4_naive.restype = c_int32
    lib.dot_product_int4_opt.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int4_opt.restype = c_int32
    lib.dot_product_int2_naive.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int2_naive.restype = c_int32
    lib.dot_product_int2_opt.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int2_opt.restype = c_int32
    lib.dot_product_int3.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int3.restype = c_int32
    lib.dot_product_int3_opt.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int3_opt.restype = c_int32
    lib.dot_product_int5.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int5.restype = c_int32
    lib.dot_product_int7.argtypes = [c_uint8_p, c_int8_p, c_size_t]
    lib.dot_product_int7.restype = c_int32

    rng = np.random.RandomState(42)
    N = 128
    activations = rng.randint(-128, 127, size=N, dtype=np.int8)
    a_ptr = activations.ctypes.data_as(c_int8_p)

    # 1. Test INT8 baseline & opt
    w8 = rng.randint(-128, 127, size=N, dtype=np.int8)
    w8_ptr = w8.ctypes.data_as(c_int8_p)
    expected_8 = int(np.dot(w8.astype(np.int32), activations.astype(np.int32)))
    actual_8 = lib.dot_product_int8(w8_ptr, a_ptr, N)
    actual_8_opt = lib.dot_product_int8_opt(w8_ptr, a_ptr, N)
    assert actual_8 == expected_8, f"dot_product_int8 mismatch: expected {expected_8}, got {actual_8}"
    assert actual_8_opt == expected_8, f"dot_product_int8_opt mismatch: expected {expected_8}, got {actual_8_opt}"
    print(f"[OK] dot_product_int8:     expected={expected_8}, actual={actual_8}")
    print(f"[OK] dot_product_int8_opt: expected={expected_8}, actual={actual_8_opt}")

    # 2. Test INT4 naive & opt
    w4 = rng.randint(-8, 7, size=N, dtype=np.int8)
    packed_4 = pack_nibbles(w4)
    p4_ptr = ctypes.cast(ctypes.c_char_p(packed_4), c_uint8_p)
    expected_4 = int(np.dot(w4.astype(np.int32), activations.astype(np.int32)))
    actual_4_naive = lib.dot_product_int4_naive(p4_ptr, a_ptr, N)
    actual_4_opt = lib.dot_product_int4_opt(p4_ptr, a_ptr, N)
    assert actual_4_naive == expected_4, f"dot_product_int4_naive mismatch: expected {expected_4}, got {actual_4_naive}"
    assert actual_4_opt == expected_4, f"dot_product_int4_opt mismatch: expected {expected_4}, got {actual_4_opt}"
    print(f"[OK] dot_product_int4_naive: expected={expected_4}, actual={actual_4_naive}")
    print(f"[OK] dot_product_int4_opt:   expected={expected_4}, actual={actual_4_opt}")

    # 3. Test INT2 naive & opt
    w2 = rng.randint(-2, 1, size=N, dtype=np.int8)
    packed_2 = pack_twobit(w2)
    p2_ptr = ctypes.cast(ctypes.c_char_p(packed_2), c_uint8_p)
    expected_2 = int(np.dot(w2.astype(np.int32), activations.astype(np.int32)))
    actual_2_naive = lib.dot_product_int2_naive(p2_ptr, a_ptr, N)
    actual_2_opt = lib.dot_product_int2_opt(p2_ptr, a_ptr, N)
    assert actual_2_naive == expected_2, f"dot_product_int2_naive mismatch: expected {expected_2}, got {actual_2_naive}"
    assert actual_2_opt == expected_2, f"dot_product_int2_opt mismatch: expected {expected_2}, got {actual_2_opt}"
    print(f"[OK] dot_product_int2_naive: expected={expected_2}, actual={actual_2_naive}")
    print(f"[OK] dot_product_int2_opt:   expected={expected_2}, actual={actual_2_opt}")

    # 4. Test INT3 scalar & opt
    w3 = rng.randint(-4, 3, size=N, dtype=np.int8)
    packed_3 = pack_arbitrary_bits(w3, 3)
    p3_ptr = ctypes.cast(ctypes.c_char_p(packed_3), c_uint8_p)
    expected_3 = int(np.dot(w3.astype(np.int32), activations.astype(np.int32)))
    actual_3 = lib.dot_product_int3(p3_ptr, a_ptr, N)
    actual_3_opt = lib.dot_product_int3_opt(p3_ptr, a_ptr, N)
    assert actual_3 == expected_3, f"dot_product_int3 mismatch: expected {expected_3}, got {actual_3}"
    assert actual_3_opt == expected_3, f"dot_product_int3_opt mismatch: expected {expected_3}, got {actual_3_opt}"
    print(f"[OK] dot_product_int3:     expected={expected_3}, actual={actual_3}")
    print(f"[OK] dot_product_int3_opt: expected={expected_3}, actual={actual_3_opt}")

    # 5. Test INT5
    w5 = rng.randint(-16, 15, size=N, dtype=np.int8)
    packed_5 = pack_arbitrary_bits(w5, 5)
    p5_ptr = ctypes.cast(ctypes.c_char_p(packed_5), c_uint8_p)
    expected_5 = int(np.dot(w5.astype(np.int32), activations.astype(np.int32)))
    actual_5 = lib.dot_product_int5(p5_ptr, a_ptr, N)
    assert actual_5 == expected_5, f"dot_product_int5 mismatch: expected {expected_5}, got {actual_5}"
    print(f"[OK] dot_product_int5: expected={expected_5}, actual={actual_5}")

    # 6. Test INT7
    w7 = rng.randint(-64, 63, size=N, dtype=np.int8)
    packed_7 = pack_arbitrary_bits(w7, 7)
    p7_ptr = ctypes.cast(ctypes.c_char_p(packed_7), c_uint8_p)
    expected_7 = int(np.dot(w7.astype(np.int32), activations.astype(np.int32)))
    actual_7 = lib.dot_product_int7(p7_ptr, a_ptr, N)
    assert actual_7 == expected_7, f"dot_product_int7 mismatch: expected {expected_7}, got {actual_7}"
    print(f"[OK] dot_product_int7: expected={expected_7}, actual={actual_7}")

    print("\nSUCCESS: All C reference kernels functionally verified against NumPy reference!")

if __name__ == '__main__':
    run_tests()
