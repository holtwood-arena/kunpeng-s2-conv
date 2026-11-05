# 公开实现调研（2026-11-05）

## 结论先行

1. **S1/S2 的获奖源码确实已公开**，但不在官方仓的 `main` 分支，而在 **37 个未合并的 Pull Request** 的源分支里。
2. **S2 CONV 至少有 8 支队伍公开了实现**，含核芯、False Sharing、kunUltra、SuperScale 等。
3. **分水岭是 SME（Scalable Matrix Extension），不是常规优化技巧**：头部方案用 SME 外积 + ZA 累加器，普通 NEON 峰值与之相差约一个数量级。
4. 因此：**这些代码是学习资料与天花板参照，不是可交付成果**。

## 官方仓库

- 仓库：<https://gitcode.com/kunpengcompute/HPC-competitions>
- 结构：
  ```
  01_official/Kunpeng_HPC_Global_Challenge_S1/submissions/
  01_official/Kunpeng_HPC_Global_Challenge_S2/submissions/
  ```
- **实测状态**：`main` 分支 HEAD 停在 2026-06-02，`submissions/` 只有 `.gitkeep`（空）。
- 用 API `GET /api/v5/repos/kunpengcompute/HPC-competitions/pulls?state=all` 核实：**共 38 个 PR，仅 #1 合并，其余 37 个全部 `open`**。

> 注意：网页上部分 PR 显示 "Merged"，与 API 的 `state=open` 不一致。以 API 与 `main` 实际内容为准。
> 拿到代码的方式是 **clone 提交者的 fork + checkout 对应分支**，而不是拉官方 main。

## S2 CONV 相关提交（8 支队伍）

| PR | 队伍 | 分支 | 备注 |
|---:|---|---|---|
| #21 | SuperScale | `superscale-conv-opensource` | 源码 + 关键优化说明 |
| #23 | 明天见 | `s2-conv-award-open-source` | CONV/ZGEMM/TRSM 三题 |
| #24 | HPC091 地球系列 | `s2-hpc091-submissions` | 三题 |
| #26 | nju | `main`（fork） | conv 算子 |
| #31 | MoneyIsAllYouNeed | `main`（fork） | 三题 |
| #32 | **False Sharing** | `codex/false-sharing-conv-e238-v3` | 获奖版本 E238 v3 |
| #34 | **核芯（hexin）** | `hexin-s2-source-submissions` | 三题，CONV 版本 D236/N300 |
| #36 | kunUltra | `kunultra-s2-submissions` | 三题 |

克隆示例（只取需要的分支）：

```bash
git clone --depth 1 --branch hexin-s2-source-submissions \
  https://gitcode.com/Swaggerer/HPC-competitions.git hexin
```

## 已实读的两份（质量评估）

### 核芯队（PR #34）

- `code/conv2d.c` **9,223 行**
- 技术要点（摘自其 README / optimization-notes）：
  - **SME FP32 外积 + ZA 累加器**，一次算多行多列输出
  - 输入**滑动窗口复用**，相邻输出共享载入
  - 卷积核按行相位**重新打包**，匹配 SME 行向计算
  - 部分核形状用 **SVE gather** 构造连续系数向量
  - 按核高/对齐/向量长度分发完整块、混合块、剩余行三套路径
  - OpenMP 划分 + 线程局部面板 + 并行触页 + **NUMA/HBM 绑定**
- 环境要求：HPCKit 25.2.0 / BiSheng clang 19.1.7，`-O3 -fopenmp`，需 SVE **且** SME
- 其自述历史总分：**54,287.98**（四案 GFLOPS 合计，两轮均值）；自述「不等同官方最终成绩」

### False Sharing 队（PR #32）

- `code/conv2d.c` **33,411 行**，且**不是单文件 C**——是一个"源码容器"：
  内含多份 C、汇编和调度脚本，由 `run.sh` 按标记提取、校验 SHA-256 后分别编译
- 技术要点：
  - 四案**分别特化**（内部版本号 E225/E235/E155/E196/cap78）
  - SVE + **SME 512-bit**，SME 部分直接用汇编与 `.inst` 编码
  - 包内 launcher 做 CPU/内存绑定与资源校验
  - 编译选项 `-O3 -fopenmp -march=armv8.2-a+sve -msve-vector-bits=512`
- 其 `results/normal.json` 记录了完整的选择流程（AB-BA 顺序、两轮重选、原子 checkpoint、270 s deadline）

## 第三份参照：`Parrotools/kunpengS2`（GitHub，非官方）

这是**最有教学价值**的一份，因为它完整记录了失败：

| 实验 | 内容 | 结果 |
|---|---|---|
| C001 | 局部标量累加器 + 删 memset | 四案全 PASS，中位数 **+14% ~ +38%** |
| C002 | 指针替代二维下标 | 四案全变慢，**REJECT** |
| C003 | 4 输出 SIMD | 四案正收益，保留 |
| C004 | 8 输出 SIMD | 本地 PASS，**鲲鹏 Case1 精度失败** |
| C005 | 核尺寸特化 | 收益混杂，回滚 |
| C007-a | double 累加器 | **鲲鹏误差更大**，回滚 |
| 最终 | 退回 baseline（C000） | 为避免精度失败 |

它同时给出 ZGEMM 的 +105% 与 TRSM 直接调 KBLAS `dtrsm` 的路线。

## 对本题的启示

1. **不要以获奖代码为起点**。9 千到 3 万行的 SME/汇编，读懂成本远超收益，且无法在 x86 验证。
2. **精度是隐藏杀手**。容差 1e-5 对 6561 项 FP32 累加并不宽松，SIMD 归约顺序变化足以翻车。
3. **分阶段、每步全量对拍**，是唯一安全的推进方式。
4. 你的项目价值不在"打赢他们"，而在**能否解释清楚：瓶颈在哪、为什么选这个方案、代价是什么**。

## 使用边界

- 本仓库**不复制**任何参赛队伍源码，只保留链接与技术要点。
- 若引用其数据，必须标注来源与"非本人实测"。
- 面试中若提及这些方案，应表述为「我调研过公开获奖实现」，**不得暗示为自己成果**。
