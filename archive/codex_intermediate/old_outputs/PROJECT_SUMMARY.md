# Limit Order Book Simulation - Output Summary

本目录包含一次 500 步 limit order book 仿真实验结果。

## 文件

- `events.csv`: 每一步的订单簿状态，包括 best bid、best ask、mid price、spread、五档深度和 imbalance。
- `trades.csv`: 成交明细，包括成交价格、数量、买方、卖方和主动方。
- `summary.json`: 汇总指标和博士课题特色配置。
- `features.csv`: 每一步的 microprice、queue imbalance、order-flow imbalance 等高频微观结构特征。
- `research_sweep.csv`: 多情景、多随机种子的实验结果。
- `thesis_transfer_sweep.csv`: 以博士论文机制为线索的迁移实验结果。
- `THESIS_TRANSFER_SUMMARY.md`: 论文机制到市场微观结构的映射和聚合结果。

## 当前博士课题特色层

当前版本以博士论文为准，把课题特色设计成从 attosecond FEL 数值研究转向高频量化的研究假设模块：

- 博士论文：`Numerical Study of the Attosecond Free-Electron Laser Pulse Generation in the Soft X-ray Regime at the SwissFEL`。
- 冲击窗口：第 180 到第 260 步。
- 论文机制：SASE/ESASE、slicing、mode-locking、TGU、slippage、superradiance、undulator taper。
- 市场机制：短暂信息冲击叠加适应性流动性补充。
- 主要指标：spread、depth imbalance、microprice deviation、order-flow imbalance、realized volatility、resilience score。
- 默认假设：论文中的随机噪声启动、延迟控制、taper 优化和 seed 鲁棒性验证流程，可以迁移到订单簿冲击、盘口恢复和队列动态研究。

如果要贴合你的真实博士课题，建议下一步替换 `configs/phd_profile.json` 中的 `doctoral_angle`、`mechanism` 和 `main_outcomes`，再添加相应 agent 或 metric。

## Demo 指标

本次默认 demo 的核心结果保存在 `summary.json`。可用以下命令重新生成：

```bash
python3 scripts/run_demo.py
```

多情景实验：

```bash
python3 scripts/run_research_sweep.py
```

以博士论文机制为线索的迁移实验：

```bash
python3 scripts/run_thesis_transfer_experiment.py
```
