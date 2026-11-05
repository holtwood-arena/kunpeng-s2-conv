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
