# 实验记录

一次实验一个文件，命名 `E##-<slug>.md`，模板见 [`../04-experiment-protocol.md`](../04-experiment-protocol.md)。

| 编号 | 标题 | 判决 |
|---|---|---|
| [E01](E01-local-accumulator.md) | 局部累加器 + 删 memset | **KEEP**（+7.9~10.7%，需关 FMA 收缩） |
| [E02](E02-row-pointer.md) | 行基址指针（对照实验） | **REJECT**（+0.2%，噪声内） |
| E03 | 循环顺序 | 待做 |
| [E04](E04-output-blocking.md) | 输出寄存器分块 4 路 | **KEEP**（×3.0~3.4） |
| [E05](E05-restrict.md) | `__restrict` 向量化 | **REJECT**（−3.0%） |
| E06 | NEON 4 输出 | 待做 |
| [E07](E07-blocking8.md) | 8 路分块（高风险） | **REJECT**（−19.2%） |
| [E08](E08-omp-schedule.md) | OpenMP 调度策略 | **REJECT**（−0.1%，噪声内） |
| [E09](E09-block2x4.md) | 2 行 × 4 列分块 | **REJECT**（−3%，噪声内） |
| E10 | 算法级（可选） | 不做（容差制约） |
| [E12](E12-output-dimension-vectorization.md) | **输出维度向量化** | **KEEP**（×2.5，当前最优） |
