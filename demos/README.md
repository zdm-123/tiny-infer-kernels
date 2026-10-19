# End-to-end demo

两个脚本都用随机权重，不下载 checkpoint。

`bench_layer.py` 是单个 block（hidden 1024，8 heads，没有分组注意力）。

`bench_generate.py` 是 Qwen2.5-0.5B 的整网形状：hidden 896，14 个 query head，2 个 KV head，intermediate 4864。默认 4 层，方便先跑通。24 层才是这个模型的真实深度。

```bash
source /home/zdm/project1/pytorch/bin/activate
cd /home/zdm/project1/task/1
python demos/bench_layer.py
python demos/bench_generate.py --layers 24 --prompt 128 --new-tokens 16
```

Triton 路径使用 RMSNorm、`rotate_half` RoPE、FlashAttention / Flash-Decoding、SwiGLU。Q、K、V、O 和 MLP 的矩阵乘仍走 PyTorch。分组注意力直接读 KV head：长 decode 时一块 K/V 给同组 query head 共用，不再把 cache 复制 7 份。对比见 `benches/bench_gqa_decode.py`。
