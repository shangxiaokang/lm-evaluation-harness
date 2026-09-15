# Server startup and memory summary

Memory breakdown values are per-worker maxima from `/server_info`.

| Mode | Quantization | Ready s | Load weight s | Weight GB | KV GB | Token capacity | Peak HBM GiB (sum) |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | None | 2192.00 | 115.00 | 117.635 | 36.524 | 3003712 | 164.886 |
| nvfp4_online | nvfp4_online | 378.00 | 229.77 | 52.908 | 101.249 | 8326784 | 164.884 |
| nvfp4_offline | modelopt_fp4 | 294.00 | 135.80 | 41.020 | 113.140 | 9304704 | 164.950 |
