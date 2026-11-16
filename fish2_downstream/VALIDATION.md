# 已完成的验证

在113上以现成Fusion50 epoch050 Balanced1152和Handcrafted22为输入，运行新入口；没有GPU提取或encoder训练。

- 5项关键测试通过：ID乱序映射、缺失/错误body拒绝、重复ID与元数据特征拒绝、同body多类型拒绝、并行输出隔离及checkpoint仅计划不执行。
- inspect核对compartment 5059实例/81 bodies、cell_type 5568实例/81 bodies。
- 新RF入口复现两条历史任务；新LR入口复现compartment。
- 比较BA和Macro-F1容差为1e-12；混淆矩阵计数完全一致。

| 协议/聚合评价 | Handcrafted BA / Macro-F1 | Balanced BA / Macro-F1 |
|---|---|---|
| compartment RF，161 oracle组 | 0.869676 / 0.869545 | 0.882176 / 0.881915 |
| compartment LR，161 oracle组 | 0.863426 / 0.863349 | 0.913194 / 0.913013 |
| cell_type RF，81 bodies | 0.531566 / 0.515942 | 0.659091 / 0.619045 |

验证运行的分类代码提交为9b35e2b；后续提交只补充记录、导出spec预检和文档，不改变拟合/划分/聚合代码。
具体日志与OOF位置：

- RF：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T122756Z_codex_validation_historical_rf_e8448c2f/
- LR：/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/runs/20261114T122755Z_codex_validation_historical_lr_985b534f/

checkpoint桥接只验证了计划与不执行路径。实际GPU导出需提供与新模型匹配、单独smoke过的原导出器；示例JSON不是已验证架构适配器。

## Cell Type LR增量验证（2026-11-14）

新增六类混合LR入口，7项单元测试通过，包含训练折scaler输入核对及禁止按真实类型训练。Handcrafted22、Balanced1152、Fused384共15次LR拟合全部收敛（129–1109次迭代）。
Compartment LR实际重跑仍复现原Balanced结果；Fused Cell Type RF实际重跑复现历史body结果。三组RF与LR的冻结实例、fold及指标复算检查全部通过。详细运行目录和RF来源见Cell Type基准报告。
