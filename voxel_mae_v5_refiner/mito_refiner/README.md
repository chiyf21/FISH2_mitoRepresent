# MAE-v5 Refiner

当前说明与完整命令见 [REFINER_ADDON_ZH.md](../REFINER_ADDON_ZH.md)。

- `train_v5.py` / `train.py`：Stage A固定特征训练Head，Stage B独立NNA精修；默认真实batch，可选Replay。
- `v5_adapter.py`：checkpoint加载与B4/B5/B6/final全局特征接口。
- `data_utils.py`：Encoder-only完整输入、双Global Views及成对跳过。
- `training_step.py`：共享Head/NNA逻辑、两种encoder内存策略、显式参数冻结。
- `evaluate.py`：完整、无噪声输入为主embedding；增强view稳定性单独比较。
- `tests/test_v5_integration.py --interfaces-only`：不运行backward或optimizer的接口检查。

当前等待用户通知后再训练/实验。`IMPLEMENTATION_REPORT_zh.md` 是历史v1报告。
旧 `run_smoke.sh` 已改为输出说明，不再自动执行旧v1训练命令。
