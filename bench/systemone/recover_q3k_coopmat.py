"""Reconstruct a Q3_K coopmat candidate from the existing Q4_K kernel.
This generates shader source only. It does not enable/register the unvalidated path.
"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
shaders = root / 'gpu' / 'shaders'
q4 = (shaders / 'gemm_q4k_coopmat.comp').read_text()
q3 = (shaders / 'gemm_q3k_tiled.comp').read_text()
q4_struct = 'struct block_q4_k {\n    float16_t d;\n    float16_t dmin;\n    uint8_t scales[12];\n    uint8_t qs[128];\n};'
q3_struct = q3.split('struct block_q3_k {', 1)[1].split('};', 1)[0]
q4 = q4.replace(q4_struct, 'struct block_q3_k {' + q3_struct + '};')
q4 = q4.replace('block_q4_k data_a[]', 'block_q3_k data_a[]')
# Preserve the known-working cooperative matrix staging/load/store implementation.
start = q4.index('// Extract 6-bit scale')
end = q4.index('void main()', start)
functions = q3[q3.index('void extract_scales('):q3.index('void main()')]
functions = functions.replace('data_b[', 'data_a[')
q4 = q4[:start] + functions + '\n' + q4[end:]
start = q4.index('                // Which Q4_K block')
end = q4.index('\n            }\n            smA[', start)
q4 = q4[:start] + '''                val.x = float16_t(dequant_q3k(m_global, k_global, K));
                val.y = (k_global + 1u < K) ? float16_t(dequant_q3k(m_global, k_global + 1u, K)) : float16_t(0.0);
''' + q4[end:]
q4 = q4.replace('Q4_K', 'Q3_K').replace('142 bytes', '110 bytes')
q4 = q4.replace('BK=32 aligns perfectly with one Q3_K scale group.',
                'BK=32 covers two Q3_K scale groups; source dequantization matches the CPU reference.')
output = shaders / 'gemm_q3k_coopmat_recovered.comp'
if output.exists():
    raise RuntimeError('Refusing to overwrite an existing recovery candidate')
output.write_text(q4)
print(f'Generated {output}; unregistered, disabled until numerical validation')
