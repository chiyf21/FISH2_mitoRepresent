# Fish2 下游任务统一入口

这里固定两条已经跑过的任务，让同用 cyf 账号的人通过同一入口查询来源、导入新 embedding、复用合法划分并运行。代码纳入 FISH2_mitoRepresent；数据、模型、结果放在仓库外。入口不需要切换共享 main。

共享路径：`/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream/`

- `run.py`：指向 Git 工作树入口的符号链接。
- `README.md`、`AGENTS.md`：本说明和给后续 Codex 的约定。
- `assets/reviewed81_v1/`：固定实例、标签、五折、22D baseline 和现成 epoch050 参考向量。
- `runs/`：每次调用自己的 owner/name/时间/随机后缀目录；不覆盖别人的结果。
- `exports/`：新 checkpoint 导出计划、日志、输出 descriptor。

## 两条任务的区别

| 协议 | compartment | cell_type |
|---|---|---|
| 样本 | 5059实例 / 81 bodies | 全部5568实例 / 81 bodies |
| 标签 | axon / dendrite，且 compartment_qc=ok | CON / GC / MON1 / MON2 / PC / VMN |
| 划分 | 固定五折，按body隔离，seed=20261113 | 固定五折，按body隔离，seed=20261113 |
| RF训练范围 | 每种神经元类型单独训练 | 六种类型混合训练一个六分类器 |
| 聚合 | body_id + true compartment，oracle grouping | body_id，每个body一个细胞类型预测 |
| 分类器 | RF；固定LR profile | RF；固定多分类LR profile |
| RF参数 | 100树、min_weight_fraction_leaf=.01、max_features=sqrt、训练类别计数倒数sample_weight；不标准化 | 相同 |
| LR参数 | L2、lbfgs、C=1、class_weight=balanced、max_iter=3000；每个类型/训练折单独StandardScaler | 同一L2/lbfgs/C=1/balanced/max_iter=3000；每折全部训练body拟合StandardScaler，多分类混合训练 |

两者均先输出实例OOF概率，聚合采用组内平均 log10(clip(p,1e-15,1)) 后 argmax。
body ID不是线粒体ID。所有embedding都以完整record_id一一连接，缺目标实例、重复ID、body不符或输入hash变化会报错，不会静默取交集。
cell_type不能按真实类型分别训练，否则已经给出了要预测的答案。

## 查询及运行

在113上执行：
```bash
FISH2_PY=/home/cyf/miniconda3/envs/cellmap_organelle/bin/python
FISH2_TASKS=/mnt/ddf22602-0d4f-4bca-aecb-6484c45f5811/cyf/EM/fish2_downstream
$FISH2_PY $FISH2_TASKS/run.py tasks
$FISH2_PY $FISH2_TASKS/run.py inspect --embedding $FISH2_TASKS/assets/reviewed81_v1/canonical_balanced1152.json
$FISH2_PY $FISH2_TASKS/run.py run --embedding $FISH2_TASKS/assets/reviewed81_v1/canonical_balanced1152.json --task all --classifier rf --with-handcrafted --owner YOUR_NAME --name fusion50_reference
$FISH2_PY $FISH2_TASKS/run.py runs
```

compartment的LR入口：
```bash
$FISH2_PY $FISH2_TASKS/run.py run --embedding $FISH2_TASKS/assets/reviewed81_v1/canonical_balanced1152.json --task compartment --classifier lr --with-handcrafted --owner YOUR_NAME --name fusion50_lr
```

`--owner`填写实际使用者，不要三个人都写cyf。`--name`填写本次实验标识；输出目录自动唯一化。`--workers`默认2；BLAS线程默认1。以上评测都在CPU上，不启动编码器或占用GPU。

Cell Type 的固定LR入口（相同实例/fold，按body聚合）：
```bash
$FISH2_PY $FISH2_TASKS/run.py run --embedding $FISH2_TASKS/assets/reviewed81_v1/canonical_balanced1152.json --task cell_type --classifier lr --with-handcrafted --owner YOUR_NAME --name celltype_balanced_lr
$FISH2_PY $FISH2_TASKS/run.py run --embedding $FISH2_TASKS/assets/reviewed81_v1/canonical_fused384.json --task cell_type --classifier lr --owner YOUR_NAME --name celltype_fused_lr
```

## 直接拿到新embedding

NPY必须配实例映射，至少含record_id和embedding_row，建议同时有body_id：
```bash
$FISH2_PY $FISH2_TASKS/run.py register --matrix /ABS/new_vectors.npy --instance-map /ABS/instance_map.csv --representation MY_CHECKPOINT_L12_balanced_CLS --checkpoint /ABS/model.pt --output /ABS/new_embedding.json
$FISH2_PY $FISH2_TASKS/run.py inspect --embedding /ABS/new_embedding.json
$FISH2_PY $FISH2_TASKS/run.py run --embedding /ABS/new_embedding.json --task all --classifier rf --with-handcrafted --owner YOUR_NAME --name my_new_checkpoint
```

CSV使用`--matrix /ABS/embedding.csv`，默认只取`embedding_`前缀列；其他列名用`--columns f0 f1 ...`显式声明。禁止ID、标签、fold和其他元数据作为特征。descriptor记录向量阶段、shape、文件SHA256、checkpoint路径/文件stat和可选模型config。

维度不限于384或1152；必须显式命名representation。point384、EM768、balanced1152、fused384各自是不同资产，不能只看维度猜是哪一步。
当前参考balanced1152是同一Fusion50 epoch050的point384 + EM768经balance拼接；fused384是后续MLP输出。参考向量仅整理已有文件，没有重新推理。

## 只拿到新checkpoint

不同模型/预处理需要各自真实的冻结导出器；本入口不会猜网络结构，也不把通用模型权重当作embedding。
`checkpoint`接受已核实的导出器spec（见`exporter.example.json`），以argv数组调用原导出器；本包没有重写encoder forward。

```bash
# 先审查适配器与输入协议；默认只生成计划，不跑GPU。
$FISH2_PY $FISH2_TASKS/run.py checkpoint --checkpoint /ABS/new.pt --export-spec /ABS/verified_exporter.json --owner YOUR_NAME --name new_ckpt
# 对应导出器已做小样本身份/确定性smoke，且存在独占空闲卡后才执行：
$FISH2_PY $FISH2_TASKS/run.py checkpoint --checkpoint /ABS/new.pt --export-spec /ABS/verified_exporter.json --owner YOUR_NAME --name new_ckpt --execute --gpu 0
```

spec的`cwd`和`argv`必须是该模型对应的真实代码，支持`{checkpoint}`、`{export_dir}`和逻辑卡`{gpu}=0`占位符；禁止shell拼接。
导出器须生成声明的矩阵和instance_map，且覆盖固定任务的record_id。执行模式检查所选物理卡无计算进程且显存空闲；忙卡会拒绝，不自动共享。
完成后生成`embedding.json`，再调用上面的inspect/run即可。示例spec是模板，不是已验证的通用encoder预设；未知架构必须先核实适配器。此次封装没有运行GPU导出。

## 历史来源和适用边界

完整路径、源码/结果SHA256和完成证据在`history.json`，不是靠目录名猜协议。

- compartment：`compartment_classification/balanced_confusion_20261113_v1/`及`balanced_lr_20261113_v1/`，有原OOF、metrics和run.log；代码提交95cf35c / 6df5d67。
- 细胞类型：`point-MAE/runs/cell_type_fig2c_rf_20261113/run_cell_type.py`及`results.json`的reviewed_81部分。5568实例、81 bodies；该目录没有独立stdout日志，原脚本也未单独记录Git提交，已保存源码SHA256。共享账号不能据此推断实际操作者。
- 更早`fish2_reviewed_81_protocol`存在seed=0、全矩阵预标准化等其他版本，仅归档说明，不混入当前默认协议。
- 历史cell-type handcrafted21去掉的是SBI；其他21D实验可能去掉骨架长度。必须看实际列清单，不能只写“21D”。本入口baseline固定22D。
- reviewed compartment标签仍有来源追溯限制；oracle区室分组使用真实区室，不代表自动发现未知区室。
- 这是复用现有任务的可操作评测协议，不代表独立于encoder训练来源的新外部测试集。

## 运行记录和开发

每次评测保存`run.json`、`run.log`、`metrics.json`、各任务实例清单、实例/聚合OOF、每折训练记录、混淆矩阵和成功/失败标记。
记录owner、完整命令、代码commit及文件hash、输入descriptor、任务资产hash、软件版本和checkpoint身份，便于三个人追溯。
`runs`只读列出各自目录，不维护容易产生并发覆盖的共享结果表。

如更换标签、样本或fold，请生成新资产版本和任务分支，不能覆盖reviewed81_v1。只换embedding时直接register/inspect/run。
代码变更遵循项目git-collaboration；不要在这个供大家调用的工作树里同时编辑。各人在自己的worktree开发，经PR集成后再更新共享入口。

测试：`python -m unittest fish2_downstream.test_controls -v`。
历史回归：`python -m fish2_downstream.verify_reference --rf-run /ABS/completed_rf_run --lr-run /ABS/completed_lr_run`。

已完成的逐项回归验证见 [VALIDATION.md](VALIDATION.md)。

Cell Type三种表示的固定LR/RF基准见 [CELL_TYPE_LR_BENCHMARK_20261114.md](CELL_TYPE_LR_BENCHMARK_20261114.md)。
