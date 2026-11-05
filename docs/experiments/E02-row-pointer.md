# E02 行基址指针替代二维下标（预期失败的对照实验）

- 日期：2026-11-05
- 上一版：`src/conv2d_v1.c`（E01，KEEP）
- 本版：`src/conv2d_v6.c`
- 编译：`gcc -O3 -fopenmp -march=native -ffp-contract=off`
- 环境：`LOCAL_REFERENCE` — Ryzen 7 5800H（6C/12T），`OMP_NUM_THREADS=6`，`taskset -c 0-5`，预热 1 次
- 采集方式：`scripts/ab_run.py` 交错 A/B（v1 与 v6 轮流跑同一用例，抵消频率漂移），每案 3 次

## 1. 观察

v1 的内层是二维下标：

```c
for (jk) for (ik)
    sum += input[(j + jk) * inputWidth + (i + ik)] * kernel[jk * kernelWidth + ik];
```

每个 `(jk, ik)` 迭代都要重算一次 `(j+jk)*inputWidth` 和 `jk*kernelWidth`。
直觉上这是可以省掉的整数乘法与地址重算。

## 2. 假设

> 二维下标里的乘法与地址重算是可观开销，显式把行基址提到 `jk` 循环外、
> 用行内偏移寻址，能降低它。

## 3. 预测与证伪

- 预测：**0 ~ +5%**（本来就不指望大收益，这是设计上的对照实验）
- 证伪条件：变化落在噪声带内，或任一变体变慢

## 4. 改动

只改寻址方式，其他一切不动：

```c
#pragma omp parallel for
for (CONVINT j = 0; j < outputHeight; ++j) {
    const CONVFLOAT* in_base = input + (size_t)j * (size_t)inputWidth;
    CONVFLOAT* out_row = output + (size_t)j * (size_t)outputWidth;
    for (CONVINT i = 0; i < outputWidth; ++i) {
        CONVFLOAT sum = 0.0f;
        for (CONVINT jk = 0; jk < kernelHeight; ++jk) {
            const CONVFLOAT* in_row = in_base + (size_t)jk * (size_t)inputWidth;
            const CONVFLOAT* k_row = kernel + (size_t)jk * (size_t)kernelWidth;
            for (CONVINT ik = 0; ik < kernelWidth; ++ik) {
                sum += in_row[i + ik] * k_row[ik];
            }
        }
        out_row[i] = sum;
    }
}
```

累加器位置、循环顺序、`float` 累加链、写回时机与 v1 完全一致 —— **只动寻址**，
这样测出来的差异才能归因到寻址方式本身。

## 5. 正确性

四案全部 PASS，最大绝对误差 `0.00e+00`。

累加顺序未变，且 `-ffp-contract=off` 保证不触发 FMA 收缩。

## 6. 性能

中位数（3 次，GFLOPS）：

| 用例 | v1（二维下标） | v6（行基址） | 变化 |
|---|---:|---:|---:|
| 1（4096×6144 k39） | 14.9764 | 14.9566 | **−0.13%** |
| 2（6144×4096 k41） | 14.9978 | 15.0219 | **+0.16%** |
| 3（4256×6390 k55） | 14.8551 | 14.8787 | **+0.16%** |
| 4（6390×4256 k81） | 14.7732 | 14.8288 | **+0.38%** |

抖动（max−min）/median：

| 用例 | v1 | v6 |
|---|---:|---:|
| 1 | 0.15% | 0.36% |
| 2 | 1.51% | 0.55% |
| 3 | 0.42% | 0.99% |
| 4 | 0.66% | 0.45% |

四个用例的变化幅度（−0.13% ~ +0.38%）与抖动（0.15% ~ 1.51%）同量级，
**方向还不一致**：两个用例略微变慢，两个略微变快。这正是噪声的典型特征。

原始数据见 `bench/results/E02-20261105-x86.csv`（24 行）。

## 7. 判决

- 结果：**REJECT（无收益）**
- 理由：变化量小于抖动，方向不一致，不满足「超过 2 倍抖动」的判定门槛
- 结论：`-O3` 下 GCC 已经完成了强度削减与地址公共子表达式消除，
  手写行基址指针在这段代码上是**零收益**操作

## 8. 为什么这个实验值得做

**它是一个主动设计成「预期失败」的对照实验。**

做性能优化最容易犯的错误，是凭直觉列出一堆「应该会更快」的点，
然后逐个试、把噪声当成收益、把回退当成失败。这个实验的作用是校准判断力：

- 先写下预测（0~5%），事后才能判断自己是不是在编故事
- 用交错 A/B + 3 次重复，让噪声可见
- 结果落在噪声里时，**承认测不出来**，而不是挑一个好看的数

同类现象在公开项目中也有记录（`Parrotools/kunpengS2` 的 C002 实验报告四案变慢）。
**那是别人的数据，我只作为背景引用；这里给出的是我自己独立复现的结果**，
两者结论一致：现代编译器早就做完了这件事。

## 9. 对后续实验的影响

- E04 的输出分块实现里也用了行基址指针，但收益来自**分块与核系数复用**，不是指针本身。
  这个实验正好排除了「指针带来收益」这个混淆解释。
- 后续不再在寻址方式上投入实验预算。
