# 赛题事实

## 来源

- 赛题页：鲲鹏高性能计算全球挑战赛（S2 赛季）
  <https://www.hikunpeng.com/developer/contests/details/bbb369de3db64a8eac4c84731f65fd53>
- 原始代码包（官方公开下载）：
  - CONV  <https://public-download.obs.cn-east-2.myhuaweicloud.com/kunpeng/conv.zip>
  - TRSM  <https://public-download.obs.cn-east-2.myhuaweicloud.com/kunpeng/trsm.zip>
  - ZGEMM <https://public-download.obs.cn-east-2.myhuaweicloud.com/kunpeng/zgemm.zip>
- 浙大 HPC101 课程页（题目与提交规范一致）：
  <https://hpc101.zjusct.io/lab/Final-Project/>

## 赛程

| 项 | 值 |
|---|---|
| 报名/提交截止 | 2026-09-15 12:00（**已截止**） |
| 平台 | 鲲鹏 920F，单 NUMA，38 核 |
| 运行约束 | `OMP_NUM_THREADS=38 numactl -N 1` |

## 计算定义

行优先连续 FP32，单通道，步长 1，不补边，**卷积核不翻转**（即二维互相关）：

```
output[y, x] = Σ_{ky=0..KH-1} Σ_{kx=0..KW-1} input[y+ky, x+kx] * kernel[ky, kx]
OH = H - KH + 1
OW = W - KW + 1
```

名义性能：`2 × OH × OW × KH × KW / 时间`（GFLOPS）。

## 官方用例

| # | 输入 H×W | 核 KH×KW | 输出 OH×OW | FLOPs（约） |
|---|---|---:|---:|---:|
| 1 | 4096 × 6144 | 39 × 39 | 4058 × 6106 | 75.4 GFLOP |
| 2 | 6144 × 4096 | 41 × 41 | 6104 × 4056 | 83.1 GFLOP |
| 3 | 4256 × 6390 | 55 × 55 | 4202 × 6336 | 161.0 GFLOP |
| 4 | 6390 × 4256 | 81 × 81 | 6310 × 4176 | 345.8 GFLOP |

> FLOPs 为按上式手算，非实测。

## 正确性

- 容差：最大绝对误差 `1e-5`
- 校验：`bench_conv.c` 内置参考实现，输出 `PASS` / `FAILED`

## 约束

- **只能修改 `conv2d.c`**；`bench_conv.c` 不可修改
- 接口签名固定：`void conv2d(const float*, int, int, const float*, int, int, float*)`

## 关键观察：算术强度

以用例 4 为例：

- FLOPs ≈ 345.8 G
- 访存 ≈ 输入 108.8 MB + 输出 105.4 MB + 核 26 KB ≈ 214 MB
- **算术强度 ≈ 1615 FLOP/Byte**

远高于任何 CPU 的 roofline 拐点（典型 10–30 FLOP/Byte），因此这是 **纯 compute-bound** 问题：
优化重心在**指令级并行与向量宽度**，而不是减少访存。
