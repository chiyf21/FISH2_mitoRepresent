# 使用 Fish2 下游任务入口

先读README.md和history.json，确认正在跑哪个任务版本。科研操作在113，经ssh cyf@166.111.72.113执行；遵循上层AGENTS.md与git-collaboration。

- 任务只含compartment和cell_type；默认不要增加研究、搜索超参数、训练encoder或改变标签/划分。
- 先tasks/inspect，再run；新embedding必须以record_id严格映射，不得用body_id充当实例ID，不得静默取子集。
- compartment默认5059/QC=ok/axon-dendrite/分类型训练/body×真实区室oracle聚合。
- cell_type默认全部5568/六类混合RF或LR/body聚合；不能按真实细胞类型拆开训练。
- 只把向量或声明的22维手工特征输入分类器，ID/标签/fold/来源等不是特征。LR的scaler只拟合对应训练折。
- owner填实际调用者，name写实验身份；输出自动唯一化，禁止覆盖别人或历史实验。
- 新checkpoint先确认真实架构、训练预处理、编码阶段和对应导出器，生成checkpoint计划；不要把示例exporter当作已验证适配器。
- 正式新导出前用原导出器做少量实例身份、shape、finite和确定性检查；仅在独占空闲GPU执行，不自动共享卡，不恢复或继续训练。
- 当前run/inspect是纯CPU；没有显式新checkpoint导出请求时，不运行GPU。
- 供多人调用的code工作树保持已提交状态；修改时另建自己的任务worktree/分支。保留既有改动，代码提交、检查、推送到确认的fork，不改共享main。
- 最后提供输出路径、实际采用的representation/checkpoint、任务版本、样本/body数、代码版本和检查结果；不能声称未做的验证已通过。
