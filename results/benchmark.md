Hardware: Tesla T4 | torch 2.14.1+cu130 | TensorRT 10.16.1.11

| Backend | Mean latency / volume (ms) | p95 latency (ms) | Speed-up vs FP32 | Mean Dice |
|---|---:|---:|---:|---:|
| pytorch_fp32 | 34476.1 | 59137.6 | 1.00x | 0.5873 |
| pytorch_amp | 14587.1 | 24866.0 | 2.36x | 0.5874 |
| tensorrt_fp16 | 6182.1 | 10212.9 | 5.58x | 0.5873 |
