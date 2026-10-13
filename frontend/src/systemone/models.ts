// The three unified System One readouts (device-any, union-trained on CPU+GPU
// features). Metrics are the measured splits-gp2 test results per engine
// (bench/systemone/RESULTS-uni-heads-2026-10-10.md).

export interface S1ModelSpec {
  id: string;
  label: string;
  params: string;
  quant: string;
  ggufPath: string;
  readoutPath: string;
  readoutName: string;
  ctx: number;
  testGpu: number;
  testCpu: number;
  eceGpu: number;
  accent: string;
  note: string;
}

const ART = 'C:\\projects\\dlgo\\systemone-train\\artifacts';
const DL = 'C:\\Users\\mohd\\Downloads';

export const S1_MODELS: S1ModelSpec[] = [
  {
    id: 'qwen35-08b',
    label: 'Qwen3.5 0.8B',
    params: '0.8B',
    quant: 'Q4_K_M',
    ggufPath: `${DL}\\Qwen3.5-0.8B-Q4_K_M.gguf`,
    readoutPath: `${ART}\\qwen35-08b-q4-readout-v10-uni.json`,
    readoutName: 'v10-uni',
    ctx: 8192,
    testGpu: 78.64,
    testCpu: 78.93,
    eceGpu: 0.0421,
    accent: '#34d399',
    note: 'Fastest. Latency engine: sub-100ms warm short decisions.',
  },
  {
    id: 'qwen38-2b',
    label: 'Qwen3.8 2B',
    params: '2B',
    quant: 'Q4_K_M',
    ggufPath: `${DL}\\Qwen3.8-2B-Q4_K_M.gguf`,
    readoutPath: `${ART}\\qwen38-2b-q4-readout-v2-uni.json`,
    readoutName: 'v2-uni',
    ctx: 8192,
    testGpu: 84.85,
    testCpu: 84.13,
    eceGpu: 0.0304,
    accent: '#818cf8',
    note: 'Balanced. ~1.2-1.3x the 0.8B latency for +6 pts accuracy.',
  },
  {
    id: 'qwen35-9b',
    label: 'Qwen3.5 9B',
    params: '9B',
    quant: 'Q3_K_M',
    ggufPath: `${DL}\\Qwen3.5-9B-Q3_K_M.gguf`,
    readoutPath: `${ART}\\qwen35-9b-q3-readout-v2-uni.json`,
    readoutName: 'v2-uni',
    ctx: 8192,
    testGpu: 93.65,
    testCpu: 93.51,
    eceGpu: 0.0365,
    accent: '#f59e0b',
    note: 'Accuracy engine: +8.5 pts over 2B, ~10x latency. Long-doc cold is slow.',
  },
];

/** The trained readouts hold at most 5 options; wider requests degrade. */
export const MAX_READOUT_OPTIONS = 5;
