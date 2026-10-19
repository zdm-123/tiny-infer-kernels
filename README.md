# tiny-infer-kernels

面向 LLM 推理关键路径的 Triton 算子库。目标不是复现一遍官方教程，而是把「正确性 → 带宽/Roofline 分析 → 调优 → 接入推理」做成一个能在企业面试里讲清楚的项目。

当前进度：Module 1 到 Module 5 都已在 RTX 5070 上跑通。Module 5 是 Qwen2.5-0.5B 的结构（分组注意力、`rotate_half` RoPE、SwiGLU）。权重是随机的，还没有下载官方 checkpoint。

## 这个仓库解决什么问题

LLM 推理的时间主要花在几类算子上：RMSNorm、RoPE、FFN（含 SwiGLU）、Attention（prefill / decode）、量化 GEMM。PyTorch eager 会把这些拆成多次 HBM 往返；Triton 的价值是在 tile 级把它们融合成更少的 kernel，并在消费级 GPU 上把账算明白。

```
x, weight  ──►  fused RMSNorm  ──►  y     (少一次中间结果写回)
Q, K, V    ──►  FlashAttention ──►  O     (不物化 S×S 注意力矩阵)
decode q   ──►  split-K attn   ──►  O     (把 seq=1 的并行度打上去)
W_int8, x  ──►  W8A16 GEMM     ──►  y     (权重带宽减半)
```

## 环境（本机已就绪）

WSL Ubuntu 24.04。代码在 `/home/zdm/project1/task/1`，虚拟环境仍在 `/home/zdm/project1/pytorch`：

| 组件 | 版本 |
|------|------|
| Python venv | `/home/zdm/project1/pytorch`（3.12） |
| PyTorch | 2.14.0+cu130 |
| Triton | 3.8.0 |
| CUDA Toolkit | 13.0（`/usr/local/cuda-13.0`） |

```bash
source /home/zdm/project1/pytorch/bin/activate
cd /home/zdm/project1/task/1
python scripts/check_env.py
pip install pytest          # 跑测试需要；只需装一次
pytest tests/
python benches/bench_rmsnorm.py
python benches/bench_softmax.py
python benches/bench_rope.py
python benches/bench_swiglu.py
python benches/bench_flash_attn.py
python benches/bench_flash_decode.py
python benches/bench_w8a16.py
python demos/bench_layer.py
python demos/bench_generate.py --layers 24 --prompt 128 --new-tokens 16
```

## 目录

```
tiny_infer_kernels/
  kernels/          融合算子、FlashAttention、Flash-Decoding、W8A16
  layers/           单层 block（无 GQA）
  models/qwen2.py   Qwen2.5-0.5B 结构的整网 prefill / decode
  utils/            计时、正确性、GPU / Roofline 辅助
benches/            延迟、带宽、TFLOPS
tests/              对照 PyTorch 参考实现
docs/LEARNING.md    学习路线
demos/bench_layer.py      单层计时
demos/bench_generate.py   整网 tokens/s
```

## 算子进度

| 模块 | 算子 | 状态 | 对标 |
|------|------|------|------|
| 1 | fused RMSNorm | 可用 | `rmsnorm_ref` / `torch.compile` |
| 1 | fused softmax | 可用 | `torch.softmax` |
| 1 | RoPE（even/odd） | 可用 | `rope_ref` |
| 1 | SwiGLU `silu(gate)*up` | 可用 | `F.silu` |
| 2 | FlashAttention-2 prefill | 可用（fp16/bf16，head dim 32/64/128/256） | `scaled_dot_product_attention` |
| 3 | Flash-Decoding | 可用（q_len=1，split-K + LSE 归并） | SDPA / 本仓库 FA2 |
| 4 | W8A16 GEMM | 可用（per-channel int8） | fp16 GEMM |
| 5 | Qwen2.5-0.5B 结构整网 | 可用（随机权重，GQA + rotate_half） | eager SDPA |

## 本机实测（RTX 5070，sm12.0，11.9 GB）

RMSNorm fp16，有效带宽按「读 x + 读 weight + 写 y」估算：

| shape | PyTorch eager | torch.compile | Triton | 相对 eager |
|-------|---------------|---------------|--------|------------|
| 4096×1024 | 0.137 ms / 122 GB/s | 0.054 ms / 311 GB/s | **0.034 ms / 494 GB/s** | 4.0× |
| 4096×4096 | 1.108 ms / 61 GB/s | 0.213 ms / 315 GB/s | **0.125 ms / 537 GB/s** | 8.9× |
| 8192×4096 | 2.273 ms / 59 GB/s | 0.475 ms / 282 GB/s | **0.239 ms / 561 GB/s** | 9.5× |

Triton RMSNorm 已经吃到约 500–560 GB/s，接近这块卡的 HBM 上限，说明它是典型的 memory-bound 融合收益。fused softmax 与 `torch.softmax` 几乎打平（都已经是融合实现）——面试里要主动讲这一点，而不是硬说 Triton 更快。

RoPE fp16，有效带宽按「读 x + 读 cos + 读 sin + 写 y」估算（cos 和 sin 合起来相当于再读一份 x）：

| shape | PyTorch eager | torch.compile | Triton |
|-------|---------------|---------------|--------|
| 8192×128 | 0.040 ms / 158 GB/s | 0.033 ms / 190 GB/s | **0.020 ms / 318 GB/s** |
| 32768×128 | 0.128 ms / 197 GB/s | 0.089 ms / 282 GB/s | **0.065 ms / 389 GB/s** |

RoPE 没有打到 RMSNorm 那档带宽：偶/奇是隔一个元素取的，warp 的加载不连续。head dim 只有 128，算子本身很小，融合的收益主要是少几次 kernel 启动。

SwiGLU fp16，有效带宽按「读 gate + 读 up + 写 y」估算：

| shape | PyTorch eager | torch.compile | Triton |
|-------|---------------|---------------|--------|
| 4096×11008 | 0.773 ms / 350 GB/s | 1.093 ms / 248 GB/s | **0.464 ms / 583 GB/s** |
| 8192×11008 | 1.541 ms / 351 GB/s | 2.183 ms / 248 GB/s | **0.921 ms / 588 GB/s** |

这里 Triton 比 `torch.compile` 更快：`silu` 和乘法被收进同一个 kernel，流量就是 3 份张量，带宽再次顶到显存上限。fp16 最大绝对误差大约 `1e-2`，和这个精度下 `silu` 的舍入同一量级。

FlashAttention-2 prefill，fp16，causal，head dim 128。TFLOPS 按 QK 和 PV 两次 matmul、causal 记一半来算：

| shape | SDPA | Triton | 相对 SDPA |
|-------|------|--------|-----------|
| B1 H8 S1024 | 0.085 ms / 25.3 TFLOPS | 0.092 ms / 23.4 TFLOPS | 92% |
| B1 H8 S2048 | 0.220 ms / 39.1 TFLOPS | 0.250 ms / 34.3 TFLOPS | 88% |
| B2 H8 S2048 | 0.373 ms / 46.0 TFLOPS | 0.425 ms / 40.4 TFLOPS | 88% |
| B1 H8 S4096 | 0.700 ms / 49.1 TFLOPS | 0.805 ms / 42.7 TFLOPS | 87% |

差的那一截主要是 SDPA 走的是厂商已经写好的 attention kernel（更激进的软件流水和 Blackwell 上的调度）。这个 Triton 版本没有用 TMA，也没有 warp specialization。正确性对 fp32 公式的最大绝对误差约 `1e-3`。

Flash-Decoding，q_len=1，fp16，head dim 128。对比的是同一条 KV cache 上的 SDPA，以及本仓库的 prefill kernel（它在 q_len=1 时几乎只发出 `batch * heads` 个 program）：

| shape | splits | SDPA | FA2 prefill | Flash-Decoding |
|-------|--------|------|-------------|----------------|
| B1 H8 Sk4096 | 12 | 0.042 ms | 0.172 ms | **0.041 ms** |
| B1 H8 Sk8192 | 12 | 0.074 ms | 0.340 ms | **0.070 ms** |
| B1 H32 Sk4096 | 3 | 0.126 ms | 0.182 ms | 0.128 ms |
| B4 H8 Sk4096 | 3 | 0.128 ms | 0.183 ms | 0.127 ms |

batch 和 head 都很小的时候，split-K 把 prefill kernel 的空转补上，延迟和 SDPA 打平，大约是自己那个 FA2 的 4 倍。head 或 batch 已经够多时，splits 自动降下来，三者接近。这就是要讲的边界：decode 缺的是并行度，不是又一份 softmax。

W8A16，N=K=4096，权重流量从 32 MB 降到 16 MB。误差是和「先反量化成 fp16 再做 GEMM」比的最大绝对误差，K 大了以后会到 `1e-1` 这一档：

| M | fp16 GEMM | W8A16 | 相对 fp16 |
|---|-----------|-------|-----------|
| 1 | 0.072 ms / 0.47 TFLOPS | **0.054 ms / 0.62 TFLOPS** | 1.33× |
| 8 | 0.072 ms / 3.75 TFLOPS | **0.059 ms / 4.55 TFLOPS** | 1.22× |
| 64 | 0.076 ms / 28.3 TFLOPS | **0.061 ms / 35.0 TFLOPS** | 1.25× |
| 256 | **0.140 ms / 61.3 TFLOPS** | 0.153 ms / 56.1 TFLOPS | 0.92× |

M 小的时候权重读是大头，int8 更快。M=256 开始更吃算力，寄存器里的反量化补不回 tensor core 的差距，fp16 GEMM 反超。

单层 decoder（hidden 1024，8 heads，intermediate 2816，随机权重）：prefill Triton 0.226 ms，eager 0.354 ms；decode Triton 0.144 ms，eager 0.309 ms。同一层换成 W8A16 后 decode 是 0.239 ms，因为这一层的 GEMM 还没大到能盖过额外 kernel。最大绝对误差约 `2e-3`。

整网是 Qwen2.5-0.5B 的形状：hidden 896，14 个 query head、2 个 KV head，intermediate 4864。RoPE 用的是 Qwen/Llama 的 `rotate_half`。投影仍是 PyTorch 的 linear。24 层、prompt 128、fp16、随机权重：

| | eager | Triton |
|--|-------|--------|
| prefill | 13.04 ms | 19.97 ms |
| decode 16 tokens | 55.1 tok/s | **86.0 tok/s** |

Prefill 更慢，是因为每一层除了 attention 还有 RMSNorm、RoPE、SwiGLU 几次启动，短 prompt 上这些开销盖过了融合收益，SDPA 又是厂商 kernel。Decode 变成 24 层都在 q_len=1 上跑，Flash-Decoding 把这个空转补上，大约 1.6 倍。两边 logits 的最大绝对误差约 `8e-5`。4 层、prompt 只有 128 时 Triton decode 优势会消失，那是启动开销，不是算法反了。

分组注意力不再把 KV cache 复制 7 份。query head 直接按 `q_head // 7` 去读原来的 KV head。上下文到 2048 以上时，一个 program 负责一个 KV head，同一块 K/V 给组里的 query head 复用。Qwen2.5-0.5B 的 head 配比（14 / 2）、fp16、head dim 64：

| KV 长度 | SDPA | 先复制 KV 再 decode | 原生 GQA |
|---------|------|---------------------|----------|
| 4096 | 0.018 ms | 0.039 ms | **0.030 ms** |
| 8192 | 0.026 ms | 0.066 ms | **0.044 ms** |

长上下文上，少掉那 7 倍的 KV 读取后，decode attention 比旧路径快约 1.3–1.5 倍。短上下文仍按 query head 并行，避免 KV head 只有 2 个时 GPU 填不满。SDPA 还是更快，差在厂商 kernel 的调度，不在多读了一遍 cache。

## 面试时怎么讲（现在就能讲的部分）

1. **RMSNorm 是 memory-bound。** 融合后 HBM 流量约为读 x + 读 weight + 写 y。加速来自少写中间结果，不是多算 FLOPs。
2. **Softmax 官方教程的 4x 账。** naive 约 `8MN+4M` 字节，fused 约 `2MN`。行太长时这块 SRAM 放不下，必须改成 online softmax —— 这就是 FlashAttention 要做的事。
3. **正确性优先于速度。** 每个 kernel 都先对 PyTorch 参考做 `allclose`，再报 GB/s。
4. **知道边界。** 消费卡没有 Hopper 的 TMA / warp specialization。Flash-Decoding 只在 batch×head 填不满 GPU 时有用。W8A16 只在小 M、权重带宽主导时有用。
5. **Decode 的并行度。** q_len=1 时 prefill kernel 的 grid 只有 batch×head。沿 KV 切分，每个 split 交回 `(m, l, acc)`，再按全局 max 重标。归并时 `acc` 还没除过 `l`，不能再乘一次 `l`。
6. **整网不要只报一个加速比。** Qwen2.5-0.5B 深度上，prefill 目前慢于 eager，decode 大约 1.6 倍。分组注意力在长上下文里按 KV head 读一次 cache，4096 token 时比复制 KV 快约 1.3 倍。

学习顺序和每周自测见 [docs/LEARNING.md](docs/LEARNING.md)。
