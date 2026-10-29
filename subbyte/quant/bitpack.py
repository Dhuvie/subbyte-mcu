"""
Sub-byte Integer Bit-Packing and Unpacking for arbitrary bit widths b in {2..8}.
Packs integer arrays into byte sequences and reconstructs them exactly.
"""

import numpy as np

def pack_bits(int_array: np.ndarray, bits: int) -> bytes:
    """
    Packs a 1D numpy array of unsigned integers in [0, 2^bits - 1] into a byte string.
    Bits are packed consecutively LSB-first into bytes.
    """
    if bits == 8:
        return int_array.astype(np.uint8).tobytes()
        
    flat = int_array.flatten().astype(np.uint64)
    total_elements = len(flat)
    if total_elements == 0:
        return b""
        
    total_bits = total_elements * bits
    total_bytes = (total_bits + 7) // 8
    out = np.zeros(total_bytes, dtype=np.uint8)
    
    current_byte_idx = 0
    bit_offset = 0
    
    for val in flat:
        val = int(val) & ((1 << bits) - 1)
        bits_remaining = bits
        while bits_remaining > 0:
            bits_can_write = 8 - bit_offset
            bits_to_write = min(bits_remaining, bits_can_write)
            mask = (1 << bits_to_write) - 1
            chunk = (val & mask) << bit_offset
            out[current_byte_idx] |= chunk
            
            val >>= bits_to_write
            bits_remaining -= bits_to_write
            bit_offset += bits_to_write
            
            if bit_offset == 8:
                bit_offset = 0
                current_byte_idx += 1
                
    return out.tobytes()

def unpack_bits(packed_bytes: bytes, bits: int, count: int) -> np.ndarray:
    """Unpacks count integers of bitwidth bits from packed_bytes."""
    if bits == 8:
        return np.frombuffer(packed_bytes[:count], dtype=np.uint8)
        
    byte_arr = np.frombuffer(packed_bytes, dtype=np.uint8)
    out = np.zeros(count, dtype=np.uint8)
    
    current_byte_idx = 0
    bit_offset = 0
    
    for i in range(count):
        val = 0
        bits_collected = 0
        while bits_collected < bits:
            if current_byte_idx >= len(byte_arr):
                break
            bits_can_read = 8 - bit_offset
            bits_to_read = min(bits - bits_collected, bits_can_read)
            mask = (1 << bits_to_read) - 1
            chunk = (int(byte_arr[current_byte_idx]) >> bit_offset) & mask
            val |= (chunk << bits_collected)
            
            bits_collected += bits_to_read
            bit_offset += bits_to_read
            
            if bit_offset == 8:
                bit_offset = 0
                current_byte_idx += 1
                
        out[i] = val
        
    return out
