"""
Physical Binary Serialization and Deserialization for .qpack Container.
Strictly implements Section IV-A (Equations 3-5) of the manuscript:
  - 64-byte Global Container Header
  - 16-byte Per-Layer Metadata Record
  - 4-byte FP32 Scales per output channel (4 * C_l bytes)
  - Dense bit-packed weight payload (ceil(N_l * b_l / 8) bytes)

Exact on-disk file size satisfies:
  Size = 64 + sum_{l=1}^L (16 + 4 * C_l + ceil(N_l * b_l / 8))
"""

import struct
import zlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

MAGIC = b"QPACK100"  # 8 bytes
GLOBAL_HEADER_SIZE = 64
LAYER_RECORD_SIZE = 16

def pack_bits(int_weights: np.ndarray, bits: int) -> bytes:
    """
    Packs discrete integer weights in [-2^(b-1), 2^(b-1)-1] into dense big-endian bitstream.
    Signed two's complement mapped to unsigned b-bit integers [0, 2^b - 1].
    """
    flat = int_weights.flatten()
    n = len(flat)
    mask = (1 << bits) - 1
    # Convert signed two's complement to unsigned
    u_vals = (flat.astype(np.int64) & mask).tolist()
    
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    
    out = bytearray(total_bytes)
    bit_pos = 0
    for val in u_vals:
        # Pack `bits` bits starting from bit_pos (big-endian bitstream)
        rem_bits = bits
        while rem_bits > 0:
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            bits_can_write = 8 - bit_offset
            bits_to_write = min(rem_bits, bits_can_write)
            
            shift = rem_bits - bits_to_write
            chunk = (val >> shift) & ((1 << bits_to_write) - 1)
            out[byte_idx] |= (chunk << (bits_can_write - bits_to_write))
            
            bit_pos += bits_to_write
            rem_bits -= bits_to_write
            
    return bytes(out)

def unpack_bits(data: bytes, bits: int, count: int) -> np.ndarray:
    """
    Unpacks dense big-endian bitstream into signed integer weights.
    """
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    out = np.zeros(count, dtype=np.int32)
    bit_pos = 0
    
    for i in range(count):
        val = 0
        rem_bits = bits
        while rem_bits > 0:
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            bits_can_read = 8 - bit_offset
            bits_to_read = min(rem_bits, bits_can_read)
            
            shift = bits_can_read - bits_to_read
            chunk = (data[byte_idx] >> shift) & ((1 << bits_to_read) - 1)
            val = (val << bits_to_read) | chunk
            
            bit_pos += bits_to_read
            rem_bits -= bits_to_read
            
        # Convert two's complement to signed
        if val & sign_bit:
            val -= (1 << bits)
        out[i] = val
        
    return out

def save_qpack_binary(
    filepath: Path,
    model_name: str,
    layer_records: List[Dict[str, Any]]
) -> int:
    """
    Serializes layers into binary .qpack format strictly satisfying Eq. 4.
    Returns total bytes written.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_layers = len(layer_records)

    # 1. Prepare layer tables and payloads
    layer_records_bytes = []
    layer_scales_bytes = []
    layer_payloads_bytes = []

    # Calculate starting payload offset: Header (64) + Layers (16 * L)
    current_offset = GLOBAL_HEADER_SIZE + LAYER_RECORD_SIZE * num_layers

    for r in layer_records:
        shape = r['shape'] # e.g. [C_out, C_in, Kh, Kw] or [C_out, C_in]
        c_out = shape[0]
        c_in = shape[1] if len(shape) > 1 else 1
        kh = shape[2] if len(shape) > 2 else 1
        kw = shape[3] if len(shape) > 3 else 1
        bits = r['bits']
        int_weights = r['int_weights'] # discrete integer tensor
        scales = r['scales'].astype(np.float32) # FP32 per-channel scale

        assert len(scales) == c_out, f"Scales count {len(scales)} != C_out {c_out}"
        packed_w = pack_bits(int_weights, bits)
        n_params = int_weights.size
        expected_w_bytes = (n_params * bits + 7) // 8
        assert len(packed_w) == expected_w_bytes
        
        padding_bits = (expected_w_bytes * 8) - (n_params * bits)

        # Scale bytes: 4 * C_out
        s_bytes = scales.tobytes()
        assert len(s_bytes) == 4 * c_out

        payload_bytes = len(s_bytes) + len(packed_w)

        # 16-byte record: c_out(H), c_in(H), kh(B), kw(B), bits(B), ndim(B), offset(I), payload_bytes(I)
        ndim = len(shape)
        record = struct.pack(
            "<HHBBBBII",
            min(65535, c_out),
            min(65535, c_in),
            min(255, kh),
            min(255, kw),
            bits,
            ndim,
            current_offset,
            payload_bytes
        )
        assert len(record) == LAYER_RECORD_SIZE

        layer_records_bytes.append(record)
        layer_scales_bytes.append(s_bytes)
        layer_payloads_bytes.append(packed_w)
        current_offset += payload_bytes

    # 2. Build 64-byte Global Header
    name_bytes = model_name.encode("ascii", errors="replace")[:16].ljust(16, b"\x00")
    # Reserve CRC placeholder
    temp_header = struct.pack(
        "<8s16sIII28s",
        MAGIC,
        name_bytes,
        num_layers,
        1,  # global_mode (1 = symmetric per-channel)
        0,  # crc32 placeholder
        b"\x00" * 28
    )
    assert len(temp_header) == GLOBAL_HEADER_SIZE

    # Compute CRC32 over all metadata and payloads
    crc = zlib.crc32(b"".join(layer_records_bytes))
    for s_b, p_b in zip(layer_scales_bytes, layer_payloads_bytes):
        crc = zlib.crc32(s_b, crc)
        crc = zlib.crc32(p_b, crc)

    global_header = struct.pack(
        "<8s16sIII28s",
        MAGIC,
        name_bytes,
        num_layers,
        1,
        crc & 0xFFFFFFFF,
        b"\x00" * 28
    )
    assert len(global_header) == GLOBAL_HEADER_SIZE

    with open(filepath, "wb") as f:
        f.write(global_header)
        for rec in layer_records_bytes:
            f.write(rec)
        for s_b, p_b in zip(layer_scales_bytes, layer_payloads_bytes):
            f.write(s_b)
            f.write(p_b)

    return filepath.stat().st_size

def load_qpack_binary(filepath: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Deserializes a binary .qpack container. Returns header info and unpacked layers.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header_bytes = f.read(GLOBAL_HEADER_SIZE)
        assert len(header_bytes) == GLOBAL_HEADER_SIZE
        magic, name_raw, num_layers, global_mode, crc32_val, _ = struct.unpack(
            "<8s16sIII28s", header_bytes
        )
        if magic != MAGIC:
            raise ValueError(f"Invalid .qpack binary magic: {magic}")

        model_name = name_raw.decode("ascii", errors="ignore").rstrip("\x00")
        
        # Read layer metadata records
        records_raw = []
        for _ in range(num_layers):
            rec_bytes = f.read(LAYER_RECORD_SIZE)
            assert len(rec_bytes) == LAYER_RECORD_SIZE
            records_raw.append(struct.unpack("<HHBBBBII", rec_bytes))

        layers_data = []
        for rec in records_raw:
            c_out, c_in, kh, kw, bits, ndim, offset, payload_bytes = rec
            f.seek(offset)
            
            # Read scales: 4 * c_out bytes
            scales_bytes = f.read(4 * c_out)
            scales = np.frombuffer(scales_bytes, dtype=np.float32)
            
            # Read packed weights
            w_bytes_len = payload_bytes - (4 * c_out)
            packed_w = f.read(w_bytes_len)
            
            # Reconstruct exact shape from ndim
            if ndim == 1:
                shape = (c_out,)
            elif ndim == 2:
                shape = (c_out, c_in)
            elif ndim == 3:
                shape = (c_out, c_in, kh)
            else:
                shape = (c_out, c_in, kh, kw)
            total_elements = int(np.prod(shape))
            
            unpacked_int = unpack_bits(packed_w, bits, total_elements).reshape(shape)
            layers_data.append({
                'shape': shape,
                'bits': bits,
                'scales': scales,
                'int_weights': unpacked_int
            })

    header_info = {
        'model_name': model_name,
        'num_layers': num_layers,
        'global_mode': global_mode,
        'crc32': crc32_val
    }
    return header_info, layers_data

def compute_qpack_formula_size(layer_shapes: List[Tuple[int, ...]], bit_allocation: List[int]) -> int:
    """Exact size according to paper Eq. 4: 64 + sum(16 + 4*C_l + ceil(N_l * b_l / 8))"""
    size = 64
    for shape, b in zip(layer_shapes, bit_allocation):
        c_l = shape[0]
        n_l = int(np.prod(shape))
        payload = (n_l * b + 7) // 8
        size += (16 + 4 * c_l + payload)
    return size
