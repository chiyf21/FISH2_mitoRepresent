"""Build Chinese report and compact delivery tables from verified actual results."""
import argparse,json,subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from common import *
LABELS={'compartment_binary':'区室二分类','compartment_three':'区室三分类','celltype_six':'细胞类型六分类',
 'volume_nm3':'ln(体积)','surface_nm2':'ln(网格面积)','sphericity':'球形度','elongation':'ln(伸长比)','surface_to_volume_nm_inv':'ln(面积/体积)'}
SHORT={'dummy':'Dummy','simple_morphology':'简单形态基线','fusion50_point_11':'Point CLS11','fusion50_point_12':'Point CLS12',
 'fusion50_image_11':'EM CLS11','fusion50_image_12':'EM CLS12+LN','fusion50_balanced_pair_11':'Pair11','fusion50_balanced_pair_12':'Pair12',
 'fusion50_fused_11':'Fused11','fusion50_fused_12':'Fused12'}
SHORT.update({'em_base_em_mean_l11_raw':'原始EM mean11','em_base_em_mean_l12_finalnorm':'原始EM mean12+LN',
 'fusion50_em_mean_l11_raw':'Fusion50 EM mean11','fusion50_em_cls_l6_raw':'Fusion50 EM CLS6',
 'fusion50_point_mean_l12_raw':'Fusion50 Point mean12','pointmae100_point_mean_l12_finalnorm':'MAE100 Point mean12+LN'})
def fmt(x,d=3):
    if x is None or (isinstance(x,float) and not np.isfinite(x)):return '—'
    if isinstance(x,(float,np.floating)):
        if x!=0 and abs(x)<10**(-d):return f'{x:.2e}'
        return f'{x:.{d}f}'
    return str(x)
def table(headers,rows):
    def cell(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(cell,headers))+' |','|'+'|'.join(['---']*len(headers))+'|']+
      ['| '+' | '.join(map(cell,r))+' |' for r in rows])
def main(out):
    out=Path(out);folder=out/'final'
    assert (out/'evaluation_complete.json').exists()
    verification=json.loads((folder/'verification.json').read_text());assert verification['status']=='PASS'
    d=pd.read_csv(folder/'scores_long.csv');registry=json.loads((out/'feature_registry.json').read_text());by={q['name']:q for q in registry}
    audit=json.loads((out/'asset_audit.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    qc=json.loads((out/'morphology_qc.json').read_text());controls=json.loads((out/'controls_audit.json').read_text())
    exposure=json.loads((out/'pretraining_exposure.json').read_text())
    coverage=json.loads((folder/'task_coverage.json').read_text());pairs=pd.read_csv(folder/'paired_differences.csv')
    def score(name,task,metric=None):
        q=d[(d.feature==name)&(d.task==task)]
        if len(q)!=1:return None
        if metric is None:metric='body_balanced_accuracy' if task=='celltype_six' else ('balanced_accuracy' if task.startswith('compartment') else 'r2')
        return q.iloc[0].get(metric)
    def perf(names,with_ad=True):
        tasks=(['compartment_binary'] if with_ad else [])+['celltype_six','volume_nm3','surface_nm2','sphericity','elongation','surface_to_volume_nm_inv']
        headers=['表示']+[LABELS[t]+(' BA' if t.startswith('compartment') else ' body BA' if t=='celltype_six' else ' R²') for t in tasks]
        return table(headers,[[SHORT.get(n,n)]+[fmt(score(n,t)) for t in tasks] for n in names])
    def contrast(a,b,t):
        q=pairs[(pairs['first']==a)&(pairs['second']==b)&(pairs.task==t)]
        if len(q)!=1:return '对应对照见完整配对表。'
        q=q.iloc[0]
        support='区间跨零，不能据此认定稳定差异' if q.ci_low<=0<=q.ci_high else '该条件差异的区间不跨零'
        return f'{SHORT.get(a,a)} → {SHORT.get(b,b)}，{LABELS[t]}：Δ={fmt(q.delta_second_minus_first,4)}，95% body bootstrap区间[{fmt(q.ci_low,4)}, {fmt(q.ci_high,4)}]；{support}。'
    # Full machine-readable unified table: all requested scalar metrics and effective test coverage.
    wide=[]
    for name,g in d.groupby('feature',sort=False):
        q=by.get(name,{});r=dict(feature=name,stage=q.get('stage','baseline'),dimension=int(g.dimension.iloc[0]))
        for _,v in g.iterrows():
            for field in ['n_test','bodies_test','balanced_accuracy','macro_f1','body_equal_balanced_accuracy',
                          'body_balanced_accuracy','body_macro_f1','r2','mae','spearman','fold_metric_mean','fold_metric_std']:
                if field in v and pd.notna(v[field]):r[v.task+'__'+field]=v[field]
        wide.append(r)
    pd.DataFrame(wide).to_csv(folder/'unified_performance.csv',index=False)
    flat=[]
    for q in registry:flat.append({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in q.items()})
    pd.DataFrame(flat).to_csv(out/'feature_registry.csv',index=False)
    coverrows=[]
    for task,folds in coverage.items():
        ids=np.concatenate([np.load(folder/'predictions'/f'dummy__{task}.npz')['cohort_index']])
        test_n=int(d.loc[d.task==task,'n_test'].iloc[0]);test_b=int(d.loc[d.task==task,'bodies_test'].iloc[0])
        def span(s,k):
            a=[f[s][k] for f in folds]
            return str(a[0]) if min(a)==max(a) else f'{min(a)}–{max(a)}'
        coverrows.append([LABELS[task],span('train','instances')+'/'+span('train','bodies'),
            span('val','instances')+'/'+span('val','bodies'),f'{test_n}/{test_b}',len(folds)])
    pd.DataFrame(coverrows,columns=['task','train_instances_bodies','validation_instances_bodies','test_instances_bodies','folds']).to_csv(folder/'coverage.csv',index=False)
    selected=['dummy','simple_morphology','fusion50_point_11','fusion50_point_12','fusion50_image_11','fusion50_image_12',
       'fusion50_balanced_pair_11','fusion50_balanced_pair_12','fusion50_fused_11','fusion50_fused_12','fusion50_point_mean_l12_raw',
       'fusion50_em_mean_l11_raw','em_base_em_mean_l11_raw','pointmae100_point_mean_l12_finalnorm','fusion50_em_cls_l6_raw']
    classified={}
    for name in selected:
        for task in ['compartment_binary','celltype_six']:
            q=json.loads((folder/f'{name}__{task}.json').read_text())
            classified[name+'__'+task]={k:v for k,v in q['metrics'].items() if k in ['class_report','confusion','body_confusion','body_class_report']}
    save(folder/'selected_confusions_and_class_metrics.json',classified)
    all_classes=[];all_matrices=[]
    task_schema=json.loads((folder/'folds.json').read_text())['tasks']
    for _,item in d[d.kind=='classification'].iterrows():
        result=json.loads((folder/f'{item.feature}__{item.task}.json').read_text());metrics=result['metrics'];classes=task_schema[item.task]['classes']
        for unit,report_key,matrix_key in [('instance','class_report','confusion'),('body','body_class_report','body_confusion')]:
            if report_key not in metrics:continue
            for c in classes:
                entry=metrics[report_key][c];all_classes.append(dict(feature=item.feature,task=item.task,unit=unit,label=c,precision=entry['precision'],recall=entry['recall'],f1=entry['f1-score'],support=entry['support']))
            all_matrices.append(dict(feature=item.feature,task=item.task,unit=unit,classes=classes,matrix=metrics[matrix_key]))
    pd.DataFrame(all_classes).to_csv(folder/'classification_by_class.csv',index=False)
    save(folder/'confusion_matrices.json',all_matrices)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S %z')+'（北京时间）'
    stages=[]
    stage_frame=pd.DataFrame(registry);stage_frame['stage']=stage_frame['stage'].str.lower()
    for stage in ['em_base','pointmae100','nna0','nna20','fusion50']:
        g=stage_frame[stage_frame.stage==stage]
        if g.empty:continue
        cps=g[['checkpoint','checkpoint_sha256']].drop_duplicates()
        stages.append([{'em_base':'原始EM-MAE','pointmae100':'Point-MAE100','nna0':'NNA0','nna20':'NNA20','fusion50':'Fusion50'}[stage],len(g),','.join(map(str,sorted(g.dimension.unique()))),
          '; '.join(Path(p).name for p in cps.checkpoint),'; '.join(h[:12] for h in cps.checkpoint_sha256)])
    pnames=['fusion50_point_group_mean_pre_pe','fusion50_point_input_mean_post_pe','fusion50_point_mean_l6_raw',
       'fusion50_point_mean_l8_raw','fusion50_point_mean_l11_raw','fusion50_point_mean_l12_raw']
    enames=['fusion50_em_patch_mean_pre_pe','fusion50_em_input_mean_post_pe','fusion50_em_mean_l6_raw','fusion50_em_mean_l8_raw',
       'fusion50_em_mean_l11_raw','fusion50_em_mean_l12_raw','fusion50_em_mean_l12_finalnorm','fusion50_em_cls_l12_raw']
    cnames=['pointmae100_point_mean_l12_raw','pointmae100_point_mean_l12_finalnorm','nna0_point_mean_l12_raw',
       'nna20_point_mean_l12_raw','fusion50_point_mean_l12_raw','nna0_point_cls_l12_raw','nna20_point_cls_l12_raw',
       'fusion50_point_12','em_base_em_cls_l11_raw','fusion50_image_11','em_base_em_cls_l12_finalnorm','fusion50_image_12',
       'em_base_em_mean_l11_raw','fusion50_em_mean_l11_raw','em_base_em_mean_l12_finalnorm','fusion50_em_mean_l12_finalnorm']
    geometries=pd.read_csv(out/'representation_geometry.csv')
    rankrows=[[SHORT.get(r.feature,r.feature),fmt(r.raw_effective_rank,2),fmt(r.top_pc_share)] for _,r in geometries.iterrows()
         if r.feature in ['fusion50_point_12','fusion50_image_11','fusion50_image_12','fusion50_fused_12']]
    scales=pd.read_csv(folder/'scale_control.csv')
    scalerows=[[SHORT.get(q.feature,q.feature),LABELS[q.target],fmt(q.volume_only_r2),fmt(q.augmented_r2),fmt(q.partial_r2_relative_to_volume)]
        for _,q in scales.iterrows()]
    typecounts=[q for q in audit['counts']['cell_type'] if q['label'] in ['CON','GC','MON1','MON2','PC','VMN']]
    text=f"""# JaneliaEM 线粒体实例表征：多层次下游线性评价

生成时间：{now}。实验目录：{out}。代码分支：{branch}，报告生成时提交：{commit}。

## 已实际完成的工作与主结果

本次复用16组 Fusion50 CLS/Pair/Fused 缓存，补充安全 hook 导出、原始 Point-MAE100、实际 NNA epoch0 初始化、NNA20、原始 EM-MAE 对照，最终得到 **{len(registry)}组实例向量**，每组固定10,678个稳定身份。这不是{len(registry)}个独立训练模型。加上2个基线，共完成 **{len(d)}个表示×任务评价单元**；首轮已有缓存评价的72个结果在统一评价中逐项复现。编码器始终冻结，没有新 MAE/NNA/Fusion 训练。

下表是核心正式结果摘录。类型列为概率按body聚合后的 Balanced Accuracy；回归在列名所示尺度评价。所有同任务表示使用相同实例、相同分组划分及同一探针代码。完整67组表示和2个基线见 [完整统一性能表](final/unified_performance.csv)，全精度长表见 [scores_long.csv](final/scores_long.csv)。

{perf(selected)}

部分冻结前端向量在不同阶段完全相同，保留它们用于身份/冻结控制，不将重复向量视作独立证据。未注明阶段的Point CLS、EM CLS、Pair、Fused行均来自Fusion50；原始EM和MAE100已单独标注。

可直接观察：融合、单模态和拼接在不同任务上没有统一排序。表中的最高测试分数是探索性观察，不是使用测试集重新选择编码器、超参数或目标。差异强弱应结合下文配对区间，而非仅看小数点排序。

本次最值得保留的观察：

- Point局部group mean已能预测球形度，R²={fmt(score('fusion50_point_group_mean_pre_pe','sphericity'))}；到L11 mean为{fmt(score('fusion50_point_mean_l11_raw','sphericity'))}。Point加入PE的对应变化很小，不能把原始局部表示视为没有有用信息。
- EM patch mean的ln(V) R²从pre-PE的{fmt(score('fusion50_em_patch_mean_pre_pe','volume_nm3'))}，到post-PE的{fmt(score('fusion50_em_input_mean_post_pe','volume_nm3'))}，再到L11 mean的{fmt(score('fusion50_em_mean_l11_raw','volume_nm3'))}。相当一部分尺度可读性在加入位置编码时已经出现。
- EM CLS11→CLS12+LN在本次形态目标上未呈现普遍退化：球形度R² {fmt(score('fusion50_image_11','sphericity'))}→{fmt(score('fusion50_image_12','sphericity'))}，ln(伸长比) {fmt(score('fusion50_image_11','elongation'))}→{fmt(score('fusion50_image_12','elongation'))}。不能仅凭raw秩从约211降到约5认定这些信息消失；本实验也没有排除既有PE扰动敏感性。
- 原始EM mean11的类型body BA为{fmt(score('em_base_em_mean_l11_raw','celltype_six'))}，Fusion50 EM mean11为{fmt(score('fusion50_em_mean_l11_raw','celltype_six'))}。固定input和pooling下的配对区间支持当前协议下的语义可读性下降；这不是单独L12一层造成的结论。
- Fusion50 Point mean12与CLS12的类型body BA分别为{fmt(score('fusion50_point_mean_l12_raw','celltype_six'))}和{fmt(score('fusion50_point_12','celltype_six'))}。同一编码器、同维度改变readout就有较大差异；最终CLS分数不能概括全部普通tokens。
- NNA0→NNA20的Point mean12球形度R²为{fmt(score('nna0_point_mean_l12_raw','sphericity'),6)}→{fmt(score('nna20_point_mean_l12_raw','sphericity'),6)}；CLS12的类型body BA为{fmt(score('nna0_point_cls_l12_raw','celltype_six'))}→{fmt(score('nna20_point_cls_l12_raw','celltype_six'))}。本次不支持明显的一致下游收益，不能据此推断NNA在所有目标上无效。

## 数据身份、标签和有效样本

训练cohort固定为10,678实例/130 body。当前canonical核实为{audit['canonical_count']}实例/{audit['canonical_bodies']} body，本次没有替换或扩展cohort。使用训练时的mask映射：9,870份实际填孔mask，加上808份audit确认无需填孔、保留原路径的mask。以source_run_id、body_id、global_instance组成的稳定复合键连接；Point缓存manifest逐条核对，不按旧global_instance或数组行号跨版本拼接。

原始104/13/13 body划分与Point-MAE训练、NNA/Fusion清单逐项一致。下表每格为“实例数/body数”；类型列的测试是5折OOF合计，其训练/验证范围随fold变化，所有fold类别支持均完整。

{table(['任务','训练','验证','测试/OOF','外层folds'],coverrows)}

区室标签来自dashboard/compartment/cohort_compartments.json及label_compartments.py。它使用神经元占据图、突触端点极性、BFS树、soma启发式、嵌套arbor处理和近邻投影，是**算法弱标签，不是独立人工GT**。覆盖4,743实例/64 body；axon2,276、dendrite1,914、neurite_other547、soma6，另有5,935 unknown。二分类只用axon/dendrite共4,190例；三分类为探索性任务，未强行加入soma。64个body中{controls['arbor_conflict_bodies']}个记录arbor_conflict。另有6个body的人工骨架覆盖索引，但引用文件在112、113上不可读，且尚未完成实例投影，不能冒充本次标签。标签代码的coarse物理尺度假设与本次mask形态尺度不一致，尚无独立坐标/标签校准；因此区室成绩只验证对现有弱标签的可读性。

细胞类型来自cohort.metadata.cell_type，沿Point缓存、源实例metadata和neurons.csv传播；body元数据核对无矛盾。29种已命名标签加unlabeled中，仅保留既有6种具有足够body支持的类别，没有任意合并：

{table(['类别','实例','独立body'],[[q['label'],q['instances'],q['bodies']] for q in typecounts])}

注意预测单位：原始EM mean11的实例级BA为{fmt(score('em_base_em_mean_l11_raw','celltype_six','balanced_accuracy'))}、Macro-F1为{fmt(score('em_base_em_mean_l11_raw','celltype_six','macro_f1'))}；body聚合后BA为{fmt(score('em_base_em_mean_l11_raw','celltype_six'))}、Macro-F1为{fmt(score('em_base_em_mean_l11_raw','celltype_six','body_macro_f1'))}。body分数利用同一神经元的多个实例概率聚合，不能读成单个线粒体分类准确率。

类型主任务覆盖4,462实例/67 body；其余稀有类别和unlabeled不纳入跨body分类主张。原104/13/13划分的类型类别不齐，因此使用按body分层的5折外层测试，每折在外层训练body内固定20%分组验证。训练按body实例数和类别body数加权；测试同时保存实例指标和body均值概率指标，防止大量线粒体的少数body主导结论。

## 形态目标与QC

体积为占据体素数×1,920 nm³；ZYX物理spacing严格为30×8×8 nm。面积使用filled-mask、外补1层零、Lewiner marching cubes、level0.5、step1、去除退化三角形，再在物理坐标计算三角面积，不使用体素暴露面计数冒充网格面积。伸长比为物理体素中心协方差最大/最小特征值比的平方根。球形度使用π^(1/3)(6V)^(2/3)/A，面积/体积为A/V；后两者与V、A存在确定性联系，不视为独立生物发现。

全量计算得到{qc['ok']}例通过严格几何QC，71例边关联拓扑检查未通过，1例仅一个Z层、三维协方差退化。主几何回归统一保守排除这72例，四个非体积几何目标使用同一10,606例；体积仍使用全部10,678例，唯一退化实例已另行核对实际体素数与metadata一致。没有球形度>1记录，所有纳入目标有限。详见 [morphology_qc.json](morphology_qc.json) 与 [mesh_qc_diagnostics.json](mesh_qc_diagnostics.json)。拓扑标记不等于独立分割错误GT；排除复杂接触实例也限制了结果覆盖面。补零只能重建观测mask边界，不能恢复采集边界外的真实结构。

体积、面积、伸长比、A/V预先固定取自然对数；球形度保持原值。R²允许为负，MAE在同一评价尺度上报告；Spearman和逐目标有效数见完整长表。没有为预期结果筛掉异常分数。已有手工形态任务的体素面计数面积与可能的NaN没有直接用于本次主目标。

## 特征资产与真实读出协议

{table(['阶段','向量数','维度','实际checkpoint','SHA256前12位'],stages)}

完整位置、pooling、LayerNorm、checkpoint SHA、输入协议、mask版本、稳定ID顺序hash及生成代码配置见 [feature_registry.csv](feature_registry.csv) 和 [feature_registry.json](feature_registry.json)。Fusion50权重为epoch50/step8050，与dashboard硬链接checkpoint一致；16组缓存的feature-key由请求、inode和IDs重新推导为b9a9cbbcd15a81c16c36。新导出Point/EM L11/L12 CLS在全部10,678行与既有缓存逐元素完全一致。Point后续初始化直接读取epoch100/step46000；验证best98不参与初始化。

GPU使用FP32 prepare、bf16 autocast原生blocks、FP32 pooling，TF32关闭。输入固定surface sample0，无随机增强，每实例固定FPS种子。Point恢复XYZ物理尺度后单位球化，64个实际group。EM按原运行代码全crop min-max、以mask选择occupied patches、最多2,048个token；背景灰度没有置零。EM空间子采样沿用实际代码的spacing=(1,1,1)体素协议，没有改成新的物理采样规则。

直接调用既有core.prepare和System/native TransformerEncoder前向；forward hooks仅汇聚，不替换Transformer运算。原代码用functional Linear绕过patch Conv模块，因此使用只读TorchFunctionMode捕获原projection输出。普通token均值排除CLS和padding；EM使用真实valid mask。Point位置编码在每个block前重复加入；EM在输入加入一次。L12 finalnorm的mean是逐token LayerNorm后再均值，区别于raw block mean。所有pooling在前向中即时完成，不保存全量海量token。

## Transformer层次与pooling对照

Point链：

{perf(pnames)}

EM链，最后一行是raw CLS补充诊断，其余为普通token均值：

{perf(enames)}

pre-PE→post-PE同时引入位置编码；post-PE→block mean反映同一mean读出下的上下文变化；同层mean→CLS改变readout，不能简单全部归因于Transformer。尤其EM CLS11为raw，既有CLS12还经过最终LayerNorm；新增raw CLS12和mean12 raw/finalnorm用于拆分这一混杂。

{contrast('fusion50_em_patch_mean_pre_pe','fusion50_em_input_mean_post_pe','volume_nm3')}

{contrast('fusion50_em_input_mean_post_pe','fusion50_em_mean_l6_raw','volume_nm3')}

{contrast('fusion50_em_mean_l11_raw','fusion50_image_11','sphericity')}

{contrast('fusion50_em_mean_l11_raw','fusion50_em_mean_l12_raw','sphericity')}

{contrast('fusion50_em_cls_l12_raw','fusion50_image_12','sphericity')}

## 训练阶段对照与NNA初始化控制

{perf(cnames)}

NNA epoch0是实际保存的训练前checkpoint；其{controls['nna0_point_tensors_identical_to_mae100']}个Point张量与MAE100相同。MAE100无CLS→NNA0含初始化CLS的差异反映上下文/readout加入；NNA0→NNA20才是同结构下NNA训练变化。NNA20→Fusion50在同Point读出协议下比较后续联合训练。冻结早期blocks不意味着早期输出绝对不变：训练后的CLS仍可通过attention改变普通token上下文。

{contrast('pointmae100_point_mean_l12_raw','nna0_point_mean_l12_raw','sphericity')}

{contrast('nna0_point_cls_l12_raw','nna20_point_cls_l12_raw','sphericity')}

{contrast('nna0_point_cls_l12_raw','nna20_point_cls_l12_raw','celltype_six')}

{contrast('nna20_point_cls_l12_raw','fusion50_point_12','sphericity')}

{contrast('nna20_point_cls_l12_raw','fusion50_point_12','celltype_six')}

{contrast('em_base_em_cls_l12_finalnorm','fusion50_image_12','sphericity')}

{contrast('em_base_em_mean_l11_raw','fusion50_em_mean_l11_raw','celltype_six')}

{contrast('em_base_em_mean_l12_finalnorm','fusion50_em_mean_l12_finalnorm','celltype_six')}

## 拼接、Fusion、低秩和尺度解释

{contrast('fusion50_image_12','fusion50_fused_12','celltype_six')}

{contrast('fusion50_point_12','fusion50_fused_12','celltype_six')}

{contrast('fusion50_balanced_pair_11','fusion50_fused_11','volume_nm3')}

{contrast('fusion50_image_12','fusion50_fused_12','sphericity')}

PairBalance使用checkpoint训练EMA，是固定正仿射变换；训练集StandardScaler会基本消除其通道尺度作用。因此Pair成绩应主要解释为同层Point+EM拼接读出，而不是证明balance本身创造了新信息。Pair维度1,152、EM768、Point/Fused384；本次没有共同PCA容量控制，不能将维度和线性容量的差异完全排除。

本次全10,678队列的raw协方差统计如下。描述性秩统计未用于探针拟合或选择：

{table(['表示','raw有效秩','PC1方差份额'],rankrows)}

有效秩高低没有直接给出下游排序。低raw方差方向仍可能承载标签相关的线性信息；“某目标线性R²降低”也不等于全部信息被删除，更不证明生物生命周期结构变化。

体积控制辅助分析将目标和embedding各通道分别在训练集对[截距,ln(V)]做残差化，仅训练残差拟合Ridge、验证选alpha，再在测试集加回体积预测。partial R²表示相对体积基线残差误差的改善，允许负值：

{table(['表示','目标','仅lnV R²','加入embedding R²','相对体积 partial R²'],scalerows)}

Point输入的二倍统一缩放测试在6个实际实例上得到完全相同的单位球点坐标；但这不意味着真实数据中的体积绝对不可预测。形状与尺度相关、体素离散化、表面采样等均可能携带间接尺寸线索。形态目标来自同一输入mask，本任务评价几何信息的线性保留，不等于独立功能标签验证。当前高体积R²是观察事实；这些机制解释仍待独立缩放/重采样干预验证，不能仅凭单位球归一化断言丢失所有尺寸信息。

## 信息减弱的可定位环节

{contrast('fusion50_point_mean_l12_raw','fusion50_point_12','celltype_six')}

{contrast('fusion50_image_11','fusion50_em_cls_l12_raw','sphericity')}

对比mean、CLS和融合读出，可定位的是本次目标的线性可读性变化。若一种CLS或融合输出下降，而同一编码器普通token mean仍保持较高分数，应优先检查readout与压缩；不能据此认定所有token已丢失信息。若raw block12与finalnorm表现不同，应将最终归一化单独作为因素。完整配对表保存所有预先登记对照，避免只挑选有利目标。

## 结论强度与限制

所有encoder冻结；StandardScaler只在每次训练集拟合。Logistic候选C={protocol['classification_C']}，Ridge alpha={protocol['ridge_alpha']}，训练内部验证选择，不用测试调参，也不在验证集重拟合。正式计算为float64、lbfgs max_iter10000/tol1e-4且要求收敛，单BLAS线程；隔离进程只分担不同表示，最多4个单线程进程。float32求解诊断、较短求解预算和双线程BLAS数值诊断单独保留，未混入正式表，候选网格未因分数改变。

简单形态分类基线使用ln(V)与三个物理bbox长度；体积回归基线只使用bbox，其余形态回归基线使用ln(V)+bbox，不输入对应目标或确定性等价变换。环境为Python {audit['software']['python']}、NumPy {audit['software']['numpy']}、PyTorch {audit['software']['torch']}、scikit-learn {audit['software']['sklearn']}、scikit-image {audit['software']['skimage']}。

不确定性采用预先登记对照和1,000次test-body配对cluster bootstrap，缺失某类的分类重采样略去。区间条件于已经拟合的探针，不包含重新训练encoder/探针的不确定性，也未校正多重比较。类型同时保存5折均值/标准差。区室二分类测试只有5个body，不能以348个实例当348个独立神经元；形态只有13个测试body。小差异及跨零区间均按探索性发现处理。

基础EM-MAE的67,415块/234 body预训练manifest覆盖本次全部104训练、13验证和13测试body。细胞类型新外层fold还包含原Point/NNA自监督训练已见body。因此本次验证的是**已有自监督模型的下游线性可读性**，不是全链条严格未见神经元、跨动物或真实生物状态泛化。标签派生、分割/填孔、采样、crop/位置编码、维度、不同readout和训练目标均限制因果解释；没有独立真实形态GT或生命周期真值。

下一步应优先验证人工区室投影和坐标尺度、对高分尺度线索做可控重采样/PE干预、保留Point/EM原生读出与concat基线检查Fusion压缩，并依据NNA0匹配对照针对性研究目标函数。先确认上述具体问题，再考虑新的大规模训练。本次没有重跑UMAP、DDRTree或生命周期树，也不以二维图形替代下游证据。

## 复现、文件和代码

[资产清单](feature_registry.csv)；[完整统一性能表](final/unified_performance.csv)；[简表](final/performance.csv)；[全部指标长表](final/scores_long.csv)；[任务覆盖](final/coverage.csv)；[配对差异及区间](final/paired_differences.csv)；[体积控制](final/scale_control.csv)；[全部逐类Precision/Recall/F1](final/classification_by_class.csv)；[全部混淆矩阵](final/confusion_matrices.json)；[独立核验](final/verification.json)。

每个单项结果在final/表示名__任务名.json，包含验证选择轨迹、正则化参数和fold样本数；分类另含逐类Precision/Recall与混淆矩阵；predictions/下的NPZ保存测试稳定键、body、目标和预测。folds.json保存完整划分；protocol.json、asset_audit.json、各阶段extraction/manifest.json及feature_registry.json给出源码/配置/权重/特征hash。数据、权重、token、缓存和结果均不提交Git。

代码位于服务器独立worktree：
{Path(__file__).resolve().parents[1]}
远端为已核对的个人仓库：[任务分支](https://github.com/chiyf21/FISH2_mitoRepresent/tree/{branch}/representation_probe)，[本次代码提交](https://github.com/chiyf21/FISH2_mitoRepresent/commit/{commit})。依README使用新的唯一输出目录重跑。任务分支可审查；未推送共享main，未合并PR。共享模型源码、mask、dashboard和训练资产保持只读，源文件hash核验通过。Mac只复制本报告和小型结果/元数据，不下载模型、完整向量或原始数据。
"""
    (out/'report_zh.md').write_text(text)
    save(out/'report_manifest.json',dict(report=str(out/'report_zh.md'),generated_at=now,code_commit=commit,
       feature_count=len(registry),evaluation_records=len(d),verification=str(folder/'verification.json')))
    print('Chinese report generated',len(text),'characters')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
