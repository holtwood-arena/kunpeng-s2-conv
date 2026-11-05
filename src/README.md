# 实现版本

按迭代保留，便于横向对比与回溯：

| 文件 | 说明 |
|---|---|
| `conv2d_v0.c` | 官方 baseline（来自 `conv.zip`，逐字节保留，供对拍） |
| `conv2d_v1.c` | **E01**：局部累加器 + 删 memset（KEEP，+7.9~10.7%，须 `-ffp-contract=off`） |
| `conv2d_v2.c` | E03/E04：循环重排 + 输出 tile（未开始） |
| `conv2d_v3.c` | E06：NEON 4 输出（未开始） |
| `conv2d_final.c` | 最终选定版本 |

约定：

- 所有版本保持**完全相同的函数签名**
- 每个版本头部注释写明：相对上一版改了什么、四案对拍结果
- 官方 `bench_conv.c` 不改动，放 `bench/` 下
