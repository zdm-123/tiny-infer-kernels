# 学习路线与自测（约 14 周）

结合本仓库。代码在 `/home/zdm/project1/task/1`，环境已装在 `/home/zdm/project1/pytorch`。每周建议 10–15 小时。

## 阶段 0 · 地基（第 1–2 周）

不写 CUDA，但必须能用自己的话解释：

- Tensor 的 `shape` / `stride` / `contiguous` 对 kernel 指针算术意味着什么
- `autograd.Function` 为什么是自定义算子接入 PyTorch 的方式
- SM、warp、occupancy、HBM vs SRAM、合并访问
- **Roofline：** 算子是 memory-bound 还是 compute-bound？融合为什么能加速 memory-bound 算子？

自测：

1. 一张 fp16 的 `[4096, 4096]` 矩阵，读一次再写一次，理论最小 HBM 流量是多少字节？
2. 峰值带宽若按 500 GB/s 算，这个拷贝的下限时间是多少？
3. 为什么 RMSNorm 再快也很难接近 tensor core 的 TFLOPS 峰值？

材料：PyTorch 官方 tensor 教程；GPU MODE（原 CUDA MODE）前 3 讲。

## 阶段 1 · Triton 入门（第 3–4 周）

官方教程顺序：

1. [vector-add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) — program / grid / `tl.load` / mask
2. [fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) — 对照本仓库 `kernels/softmax.py`
3. 本仓库 `kernels/rmsnorm.py` — 自己把注释里的访存账用实测 GB/s 验证一遍

命令：

```bash
source /home/zdm/project1/pytorch/bin/activate
cd /home/zdm/project1/task/1
python scripts/check_env.py
pytest tests/
python benches/bench_rmsnorm.py
python benches/bench_softmax.py
```

自测：

1. 为什么 `BLOCK_SIZE` 必须是 2 的幂？mask 是干什么的？
2. `num_warps` 从 4 改到 16，什么时候变快、什么时候变慢？
3. softmax 在 `N=16384` 之后为什么要拒绝启动，而不是继续加大 BLOCK？

作业（已完成，可当复习）：`kernels/rope.py` 与 `kernels/swiglu.py` 的 Triton 路径、测试和 `benches/bench_rope.py`、`benches/bench_swiglu.py`。对照 README 里的带宽表，自己复述每个 kernel 少了哪几次 HBM 往返。

## 阶段 2 · GEMM 与 Attention（第 5–7 周）

- [matmul](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：`tl.dot`、tiling、`GROUP_M` swizzle、`@triton.autotune`
- [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)：online softmax、causal、fp32 累加器
- 选读：vLLM 的 RMSNorm / RoPE / attention；[Triton 块级编程](https://arganzheng.life/triton-block-level-programming.html)

自测：

1. 为什么 GEMM 要用 L2 友好的 program 重排？
2. FlashAttention 不写 S 矩阵，省了多少字节（用 B, H, S, D 写出来）？
3. 正向要存 LSE，是为了反向少算什么？

## 阶段 3 · 项目模块（第 8–14 周）

| 周 | 模块 | 完成标准 |
|----|------|----------|
| 8 | RoPE + SwiGLU 融合 | 正确性通过；带宽对比 eager / compile |
| 9–10 | FlashAttention-2 prefill | 已有 forward：causal、head dim 32–256，对标 SDPA 约 87%+。下一步是把差距写细（流水线 / TMA），需要时再补 backward |
| 11 | Flash-Decoding | 已完成。q_len=1、小 batch 时相对本仓库 FA2 约 4×，与 SDPA 打平 |
| 12 | W8A16 GEMM | 已完成。M≤64、N=K=4096 时快于 fp16 GEMM；M=256 时反超失败，要能讲原因 |
| 13–14 | 端到端 | 已有 `models/qwen2.py` 和 `demos/bench_generate.py`。24 层 decode 约 86 tok/s，对 eager 55 tok/s。官方权重还没下载 |

## 每周交付物（养成面试习惯）

- 一段可以跑的 kernel + pytest
- 一张 shape–延迟–GB/s 表（放 `docs/results/`）
- 三句话：这个 kernel 省了什么访存、卡在 Roofline 哪一侧、和官方库还差在哪
