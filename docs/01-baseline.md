# 基线代码分析

## 源码

官方 `conv2d.c`，30 行（含 include 与类型定义）：

```c
void conv2d(const CONVFLOAT* input, CONVINT inputHeight, CONVINT inputWidth,
            const CONVFLOAT* kernel, CONVINT kernelHeight, CONVINT kernelWidth,
            CONVFLOAT* output)
{
    const CONVINT outputHeight = inputHeight - kernelHeight + 1;
    const CONVINT outputWidth  = inputWidth  - kernelWidth  + 1;
    memset(output, 0, outputHeight * outputWidth * sizeof(CONVFLOAT));
#pragma omp parallel for
    for (CONVINT j = 0; j < outputHeight; ++j) {
        for (CONVINT i = 0; i < outputWidth; ++i) {
            for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
                for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                    CONVINT inputIdx  = (j + jk) * inputWidth + (i + ik);
                    CONVINT kernelIdx = jk * kernelWidth + ik;
                    output[j * outputWidth + i] += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
    }
}
```

## 四个可直接识别的性能问题

### 1. 输出元素反复读写（read-modify-write）

`output[...] += ...` 在内层最热的循环里，每个 MAC 都要 load + store 一次输出。
输出地址在整个 1521–6561 次迭代中不变，编译器**不一定**能提到寄存器（别名分析受阻：
`input` / `kernel` / `output` 都是裸指针，可能重叠）。

→ 改法：局部 `float sum` 累加，循环结束写一次。

### 2. 每个线程重复 memcpy 语义的 memset

`memset` 在并行区之前对整个输出清零，随后每个元素被 `+=` 覆盖。
若改成局部累加后一次性赋值，`memset` 可完全删除（每个有效输出必被写一次）。

### 3. 内层访存跨步

`input[(j+jk)*inputWidth + (i+ik)]`：`ik` 连续（步长 1，好），`jk` 跨 `inputWidth`（6144 floats = 24 KB）。
对 81×81 的核，一次输出要跨越 81 行、每行 324 B → 每行都触发新的 cache line 读取，
但相邻输出 `i+1` 会复用同一批行，属于**空间局部性可被利用**的情况。

### 4. 内层无向量化

`output[...] += input[...] * kernel[...]` 的 RMW 模式 + 潜在别名，使 GCC 难以自动向量化。
`-fopt-info-vec` 通常会报告 "not vectorized: possible dependence"。

## 简单改动的量级预期

| 改动 | 预期收益 | 依据 |
|---|---|---|
| 局部累加器 + 删 memset | +10% ~ +40% | 消除 RMW 与冗余清零 |
| 指针替代二维下标 | 0 ~ ±5% | 现代编译器多数已优化；可能变慢 |
| 循环交换（jk/ik 外提） | +20% ~ +3× | 提升输入复用与 ILP |
| NEON 4/8 输出并行 | +2× ~ +5× | 向量宽度 4×FP32 |
| OpenMP 调度/绑核调优 | +10% ~ +50% | 消除伪共享与 NUMA 抖动 |

> 以上为**基于代码结构的推断**，未在本机实测。真实数据须在目标平台采集。

## 一个已知的真实教训

公开的 `kunpengS2` 项目（见 `docs/02-prior-art.md`）记录：

- C004（8 输出 SIMD）在鲲鹏平台上 **Case1 精度失败**（容差 1e-5）
- C007-a（double 累加器）在鲲鹏上 **误差更大**，被回滚

说明：**SIMD 化的累加顺序改变会直接影响 FP32 误差**，而 1e-5 容差对 6561 项累加并不宽松。
这是本题最容易翻车的地方，必须在每一步都做全量 4 案对拍。

---

## 理论峰值与可达性（L1 解析层，不需要编译）

优化前先估出「合理目标」在哪，这是判断是否做到位的锚点。

### 公式

```
理论峰值 GFLOPS = 核心数 × 频率(GHz) × 每周期 FP32 FLOP
```

本算子每次乘加都是 `a*b+c`，正好对应一条 FMA 指令（2 FLOP），因此：

- 128-bit NEON（每核 2 条 FMA）：2 × 4 = **8 FLOP/cycle**
- 256-bit AVX2（每核 2 条 FMA）：2 × 8 = **16 FLOP/cycle**

### 估算

| 平台 | 假设 | 理论峰值 | 按 60%~80% 可达 |
|---|---|---:|---:|
| 鲲鹏 920（38 核可见） | 2.6 GHz · NEON | ≈ 790 GFLOPS | 470 ~ 630 |
| Ryzen 7 5800H | 8 核 · 3.2 GHz · AVX2 | ≈ 410 GFLOPS | 245 ~ 330 |

> **以上是估算，不是实测。** 核数、频率、每周期 FMA 端口数都必须用 `lscpu` 和目标机实测确认。
> 鲲鹏 920F 的实际配置以赛题环境为准。

### 这个数怎么用

1. **判断是否值得继续**：到峰值 60%~70% 后，剩余空间多半是边界处理和调度开销，收益有限
2. **识别异常**：某版本只有峰值 5%，说明有结构性问题（未向量化、伪共享、寄存器溢出）
3. **回答「你觉得能到多少」**：给区间和依据，而不是拍脑袋

### 量级合理性检查

公开项目 `Parrotools/kunpengS2` 在 x86（Ryzen 9 8945HX，16 核）上记录：
基线（朴素四重循环）中位数约 **45 GFLOPS**。

这与「未向量化的标量四重循环」的预期一致 —— 说明该实现的优化空间确实在**一个数量级**，
而不是几个百分点。这也是本题值得做的核心原因。

> 该对照仅用于量级合理性检查，不是同平台实测，也不能外推为鲲鹏性能。
