"""
Serialization and Deserialization for the .qpack Standalone Binary Format.
Real model size = exact length of the .qpack file in bytes on disk.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

MAGIC = b"QPACK100"

def save_qpack(
    filepath: Path,
    model_name: str,
    layer_records: List[Dict[str, Any]]
) -> int:
    """
    Serializes packed quantized model into a standalone .qpack artifact.
    Returns total file size in bytes.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    payload_parts = []
    
    for r in layer_records:
        pw = r['packed_weights']
        sc = r['scales'].astype(np.float16).tobytes()
        zp = r['zero_points'].astype(np.int8).tobytes()
        b_bytes = r['biases'].astype(np.float16).tobytes() if r.get('biases') is not None else b""
        
        manifest.append({
            'name': r['name'],
            'shape': list(r['shape']),
            'bits': r['bits'],
            'weight_bytes': len(pw),
            'scale_bytes': len(sc),
            'zp_bytes': len(zp),
            'bias_bytes': len(b_bytes)
        })
        payload_parts.extend([pw, sc, zp, b_bytes])
        
    header_dict = {
        'model_name': model_name,
        'num_layers': len(manifest),
        'layers': manifest
    }
    header_bytes = json.dumps(header_dict).encode('utf-8')
    header_len = len(header_bytes)
    
    with open(filepath, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<I', header_len))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)
            
    return filepath.stat().st_size

def load_qpack(filepath: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Loads a .qpack binary file and unpacks all layers."""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        magic = f.read(8)
        if magic != MAGIC:
            raise ValueError(f"Invalid .qpack magic header: {magic}")
        header_len = struct.unpack('<I', f.read(4))[0]
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode('utf-8'))
        
        layers_data = []
        for l_meta in header['layers']:
            w_bytes = f.read(l_meta['weight_bytes'])
            s_bytes = f.read(l_meta['scale_bytes'])
            z_bytes = f.read(l_meta['zp_bytes'])
            b_bytes = f.read(l_meta['bias_bytes'])
            
            scales = np.frombuffer(s_bytes, dtype=np.float16)
            zero_points = np.frombuffer(z_bytes, dtype=np.int8)
            biases = np.frombuffer(b_bytes, dtype=np.float16) if l_meta['bias_bytes'] > 0 else None
            
            layers_data.append({
                'name': l_meta['name'],
                'shape': l_meta['shape'],
                'bits': l_meta['bits'],
                'packed_weights': w_bytes,
                'scales': scales,
                'zero_points': zero_points,
                'biases': biases
            })
            
    return header, layers_data

def compute_qpack_size(layer_metas: List[Dict[str, Any]], bits_allocation: List[int], header_overhead: int = 256) -> int:
    """Computes exact on-disk byte footprint for a candidate allocation."""
    total = header_overhead
    for meta, b in zip(layer_metas, bits_allocation):
        num_elems = meta['num_elements']
        num_ch = meta.get('num_channels', 1)
        has_bias = meta.get('has_bias', False)
        
        w_bytes = (num_elems * b + 7) // 8
        s_bytes = num_ch * 2
        z_bytes = num_ch * 1
        b_bytes = (num_ch * 2) if has_bias else 0
        total += (w_bytes + s_bytes + z_bytes + b_bytes)
    return total
