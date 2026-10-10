# SGAN-DTI 无泄露复现代码

本仓库对应论文《A Dual Channel Subgraph Attention Framework for
Drug--Target Interaction Prediction by Eliminating Meta-path Dependency》。代码
保留SGAN-DTI的双通道异质子图编码器、关系感知边嵌入以及DTP-GCN预测模块，
并在数据读取与构图阶段修复验证集和测试集DTI正边泄露问题。

> **版本范围说明：**这是经过整理、可独立运行的复现实现，不是历史训练脚本
> 的逐行最小差异副本。模型总体设计和本次报告结果所需的评价流程保持不变，
> 同时重新整理了数据加载、结果输出和可复现性检查。核心科学修正是：任何
> 留出集DTI标签均不再出现在消息传递图中。

## 核心修正

对测试折 `k`，使用 `(k+1) mod 10` 作为验证折，其余八折作为训练集，即
固定80%/10%/10%划分。构图后、消息传递前执行以下操作：

1. 同时从药物到靶点关系 `d_p` 和靶点到药物关系 `p_d` 中删除验证集、
   测试集正DTI边；
2. 断言图中剩余DTI边与训练集正边完全一致；
3. 训练、验证和测试全过程复用同一个仅含训练正边的图；
4. 早停、模型选择和分类阈值只使用验证集，测试集不参与任何选择。

非DTI辅助关系保持不变。任何边集合不一致都会在训练开始前触发异常。

## 仓库内容

- `model/`：SGAN-DTI模型核心代码；
- `utils/load_graph.py`：无泄露构图、双向删边和运行时断言；
- `utils/data_loader.py`：固定十折80%/10%/10%划分；
- `datasets/`：运行所需的最小数据文件；
- `scripts/audit_no_leakage.py`：真实数据边泄露审计；
- `scripts/run_three_datasets.sh`：三个数据集完整十折运行脚本；
- `tests/test_no_leakage.py`：小图回归测试；
- `reference_results/2026-10-09/`：已验证运行的逐折指标和预测结果。

## 安装

已验证环境为Python 3.9、PyTorch 2.3.1+CUDA 11.8和
DGL 2.4.0+CUDA 11.8：

```bash
conda create -n sgan-dti python=3.9 -y
conda activate sgan-dti
pip install -r requirements-cu118.txt
```

如果使用其他CUDA版本或CPU，请先安装相互兼容的PyTorch和DGL，再执行：

```bash
pip install -r requirements.txt
```

## 验证无泄露

```bash
python -m unittest tests.test_no_leakage

python scripts/audit_no_leakage.py \
  --dataset data_luo \
  --data-dir datasets/data_luo \
  --fold-indices 0,1,2,3,4,5,6,7,8,9
```

每折均应输出：

```text
validation_in_graph=0 test_in_graph=0
```

## 完整运行

在一张GPU上依次运行三个数据集的全部0--9折：

```bash
bash scripts/run_three_datasets.sh datasets cuda:0
```

运行结果保存在 `outputs/<dataset>/`，包括逐折指标、测试预测、验证集最优
检查点和十折汇总文件。

## 已验证结果

| 数据集 | AUC | AUPR | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Hetero-A | 0.9021 ± 0.0176 | 0.9109 ± 0.0204 | 0.8452 ± 0.0452 | 0.8185 ± 0.0328 | 0.8303 ± 0.0178 | 0.6672 ± 0.0435 |
| Hetero-B | 0.9015 ± 0.0065 | 0.9015 ± 0.0093 | 0.8500 ± 0.0195 | 0.8107 ± 0.0304 | 0.8293 ± 0.0103 | 0.6682 ± 0.0142 |
| Zeng数据集 | 0.9558 ± 0.0077 | 0.9520 ± 0.0107 | 0.8755 ± 0.0143 | 0.9225 ± 0.0191 | 0.8982 ± 0.0080 | 0.7925 ± 0.0160 |

“±”表示十个固定数据折之间的样本标准差。每折均重新固定随机种子为3407，
因此这里反映的是数据折差异，不是十个独立随机初始化的波动。

## 数据与许可证

代码采用MIT许可证。数据来源于第三方论文及生物医学数据库，不属于MIT
许可范围。公开上传前必须确认原始数据库和数据集是否允许再次分发；详细
说明见[DATASETS.md](DATASETS.md)。如果无法确认，请从公开仓库删除
`datasets/`，并改为提供合法的数据下载或预处理说明。
