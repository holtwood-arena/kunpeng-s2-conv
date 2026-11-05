# E05 `__restrict` 解锁自动向量化

- 日期：2026-11-05
- 上一版：`src/conv2d_v1.c`（E01）
- 本版：`src/conv2d_v2.c`
- 编译：`gcc -O3 -fopenmp -march=native -ffp-contract=off`
- 环境：`LOCAL_REFERENCE`，6 线程，`taskset -c 0-5`

## 假设

> v0/v1 未被向量化的主因是编译器无法证明 `input` / `kernel` / `output` 不重叠。

## 改动

只在三个指针参数上加 `__restrict`，其余逻辑一字未改。

## 结果

编译器**确实报告向量化成功**：

```
src/conv2d_v2.c:39:41: optimized: loop vectorized using 32 byte vectors
src/conv2d_v2.c:39:41: optimized: loop vectorized using 16 byte vectors
```

但性能**反而下降**：

| 用例 | v1（无 restrict） | v2（有 restrict） | 变化 |
|---|---:|---:|---:|
| 1024×768 k39 | 15.1863 | 14.7348 | **-3.0%** |

（v1 三次：15.1863 / 15.2035 / 15.1033；v2 三次：15.0657 / 14.5448 / 14.7348）

正确性：四案 PASS，误差 `0.00e+00`。

## 汇编核对

```
v2:  vmulps=2   vaddps=0   vfmadd=0
```

说明只有零星的 packed 乘法，**归约主循环并没有真正被向量化**——
`vaddps=0` 意味着水平归约没生成。

## 判决

- 结果：**REJECT**
- 理由：收益为负，且负值超过噪声
- **代价**：无（改动仅一个限定符，可无损回滚）

## 教训（重要）

**编译器的「loop vectorized」报告不等于有效向量化。**

`-fopt-info-vec` 报告的是「某个循环被向量化了」，但不告诉你
向量化后是否真的有收益、是否只是把标量循环拆成了收益极小的形式。
必须用实测数据判断，不能拿报告当结论。

这条与 E04 形成对照：真正带来 ×3 收益的不是让编译器「报告向量化」，
而是**改变数据复用结构**（4 路分块）。
