# ACDC 跨域可信分割

本目录提供八行实验的统一复评和汇总，以及 NACL、MbLS、标准 EDL 的训练入口。

| 方法键 | 检查点来源 |
|---|---|
| `baseline` | ACDC 单头 Swin 的源域最优 checkpoint，同时产生公共遮罩 |
| `nacl`、`mbls`、`edl` | 本目录 `train` 的 `model_best.pt` |
| `ml1_ace`、`devis` | ACDC 对应目录的正式 checkpoint |
| `fgrm` | ACDC 完成奖励微调后的 checkpoint |
| `ours` | ACDC 双头 checkpoint，输出 base 四类概率 |

## 环境和数据

在 Ours 根目录调用。ACDC 项目和数据目录通过 `--acdc-root` 和 `--data-root` 指定。默认使用相邻 ACDC 项目及其 `database`。采用 ACDC 原有模型、SimpleITK 预处理、MONAI 和 PyTorch 环境，额外需要 NumPy、nibabel、PyYAML。训练需要 CUDA。权重须来自受信来源。

用 ACDC 的固定 split，1.5T 训练 54 人、验证 13 人，3T 测试 33 人。每名患者包含 ED 和 ES。每方法使用独立的新输出目录，最终协议文件写出后可合表。失败的输出应保留供诊断，重试使用新目录。

## AutoDL 部署

在本地 Ours 根目录执行 `bash acdc_benchmark/sync_server.sh SSH别名`。同步后的两个代码目录位于 `/home/acdc-benchmark/`。

在服务器的 `tmux` 会话内启动整轮实验。

```bash
cd /home/acdc-benchmark/Ours
GPU=0 PYTHON_BIN=/root/miniconda3/bin/python \
  bash acdc_benchmark/run_server.sh "$(date +%Y%m%d_%H%M%S)"
```

- 数据统一读取 `/autodl-fs/data/acdc_benchmark/database`。首次运行会从服务器已有的 `/root/autodl-tmp/database` 同步，可用 `ACDC_DATA_SOURCE` 指定现有数据的实际位置。脚本检查挂载、固定划分和 400 个图像及标签文件的完整性。
- 可通过 `BASELINE_CKPT`、`ML1_ACE_CKPT`、`DEVIS_CKPT`、`FGRM_CKPT`、`OURS_CKPT` 指定已确认来源的权重。脚本会核验历史权重的模型兼容性及 FGRM 阶段，历史权重缺失或不兼容时使用原训练入口在源域补训。
- 权重、日志和表格写入 `/autodl-fs/data/acdc_benchmark/runs/<运行批次>/`。`run.log` 记录进度。八种方法分别用统一口径生成 `test_table.csv` 和 `val_table.csv`。现有批次可用 `bash acdc_benchmark/run_server.sh RUN_ID --validation-only` 排队补算统一源域验证表。
- DEviS 训练中断时，用相同的运行批次执行 `bash acdc_benchmark/run_server.sh RUN_ID --resume`。脚本复用已完成的权重和评估，从 DEViS 最后完整保存的 epoch 恢复。

## 运行

### 双头扰动与逐帧 TTA

论文第 3.3 节的正式适应评估使用 `run_chapter33.sh`，复用已训练的源域权重。每批 16 帧，先对每窗两次采样做概率融合，再在全部窗口体素上累积平均几何不确定性梯度，以 Adam（学习率 `1e-3`）更新三个零初始化 Adapter 一次。Adam 和 Adapter 状态保留给下一批。原生不确定图来自关闭扰动、恒等 Adapter 的冻结分支。旧逐帧伪标签实验保存在独立历史目录。

用独立目录同步代码，保留原实验及其冻结权重和评估遮罩。

```bash
ACDC_DEPLOY_ROOT=/home/acdc-perturbation bash acdc_benchmark/sync_server.sh SSH别名
```

在服务器的 `screen` 会话中，从新部署的 Ours 目录启动。

```bash
cd /home/acdc-perturbation/Ours
disk=/autodl-fs/data/acdc_benchmark
PYTHONPATH=/home/acdc-benchmark/python_deps PYTHON_BIN=/root/miniconda3/bin/python \
  bash acdc_benchmark/run_perturbations.sh /home/acdc-perturbation/ACDC "$disk/database" \
  "$disk/runs/acdc_sota_20261107_002359" "$disk/runs/head_perturbation_$(date +%Y%m%d_%H%M%S)"
```

脚本先核验真实数据、旧权重和固定 ECE，再串行训练三个替代组。原 dropout 权重直接作为对照。每个划分输出 8 行测量表及患者配对 AUROC 差和 95% 区间。TTA 的中心块教师和两步输出卷积更新使用 FP32，最终整帧推理沿用基准的 AMP 设置。

设 `ACDC_ROOT`、`ACDC_DATA`、`BASELINE_CKPT` 和 `OUT` 为实际路径。以下命令用 CUDA。使用其他设备时，两次推理均指定相同的 `--device` 和 AMP 设置。

先建立 val 参考。训练入口只读取 val 参考。

```bash
python -m acdc_benchmark evaluate \
  --acdc-root "$ACDC_ROOT" --data-root "$ACDC_DATA" \
  --method baseline --split val --checkpoint "$BASELINE_CKPT" \
  --output-dir "$OUT/val/baseline"
```

训练三项新对照。若需调整损失权重，先在源验证划分中比较。训练采用 ACDC 共同训练器的优化和 EMA 配方，选最高源验证 Dice，ECE 用于同分排序。

```bash
for method in nacl mbls edl; do
  python -m acdc_benchmark train \
    --acdc-root "$ACDC_ROOT" --data-root "$ACDC_DATA" \
    --method "$method" --reference "$OUT/val/baseline" \
    --output-dir "$OUT/train/$method"
done
```

源域选模完成后，以同一冻结 Swin 检查点建立 test 参考。其余各方法和 split 分别调用一次。

```bash
python -m acdc_benchmark evaluate \
  --acdc-root "$ACDC_ROOT" --data-root "$ACDC_DATA" \
  --method baseline --split test --checkpoint "$BASELINE_CKPT" \
  --output-dir "$OUT/test/baseline"

python -m acdc_benchmark evaluate \
  --acdc-root "$ACDC_ROOT" --data-root "$ACDC_DATA" \
  --method edl --split test --checkpoint "$OUT/train/edl/model_best.pt" \
  --reference "$OUT/test/baseline" --output-dir "$OUT/test/edl"

python -m acdc_benchmark table --results-dir "$OUT/test" --output "$OUT/test_table.csv"
```

既有方法可用本地 ACDC 项目原有训练入口准备权重。旧入口生成的可靠性表属于旧协议，正式八行表通过以上评估命令生成。

## 与 BraTS 第 3.3 节对齐的 ACDC 实验

在部署后的 `Ours` 目录运行 `bash acdc_benchmark/run_paper_aligned.sh ACDC_ROOT DATA_ROOT ORIGINAL_RUN OUTPUT_DIR`。`ORIGINAL_RUN` 提供原 3T 冻结 Swin 遮罩，`OUTPUT_DIR` 使用一个新目录。

该入口复用 BraTS 的有符号方向 Zernike 实现，每个体素保留 3 个尺度 × 4 类 × 10 个方向分量。五组从头训练，使用与 BraTS 对齐的源域阶段、损失和不确定性参数。ACDC 保留单模态输入、四类 softmax 和自身空间预处理。只保存用于选模的训练期验证指标，跳过训练后源域评估、Z0 拟合和 UEO 阈值拟合。每组训练结束立即运行 3T 在线 TTA，随后从同一检查点做 3T 冻结推理。最后生成 `test_table.csv` 和 `test_paired_auc.csv`，ECE 和评估遮罩沿用原协议。Native AUROC 的 U 来自恒等 Adapter 下的无扰动预测，符合第 3.3 节的不确定性评估定义。

## 指标和产物

ECE、NLL、Brier、高置信错误比例直接调用 BraTS Ours 的 `FixedECEAccumulator`。所有方法共用冻结 Swin 前景与 GT 前景的并集，并排除 ACDC padding。ECE 为四类 top-label、15 个等宽箱、全遮罩体素。每帧先计算指标，再按患者 ED／ES 等权汇总，最后按患者等权。CSV 中 ECE 保留 0–1 的原值，展示为百分比时乘 100 一次。各方法原训练器的 ECE 只用于其训练流程，统一表格由冻结权重重新推理生成。

每次评估输出 `frame_metrics.csv`、`patient_metrics.csv`、`domain_metrics.csv`、`protocol.json`。合表核对评估协议、ECE 源码、冻结参考权重身份和逐帧遮罩及标签身份。

在独立输出目录为整个划分的评估加 `--extended-metrics`，可同时输出原始物理网格上的三类 HD95（mm）和固定 ECE 遮罩上的体素 AURC。HD95 使用两方向第 95 百分位表面距离中的较大值，预测为空且 GT 存在时计入漏分帧数。AURC 按最大类别概率降序保留体素，相同置信度使用同一阈值，以新增覆盖率乘该阈值下的累计错误率求和。CSV 保留 AURC 比例原值。

追加 `--ueo` 计算固定遮罩内不确定区域与实际误分区域的 Dice。不确定性为 `1 − 最大类别概率`。1.5T 验证从 0.00 到 0.75、步长 0.01 选择患者平均 UEO 最高的阈值，同分取较大阈值。3T 测试同时传入 `--ueo-validation` 指向同方法的 1.5T `protocol.json`，复用已冻结阈值。两个区域均为空时 UEO 取 1。

追加 `--error-auroc` 计算体素错误检测 AUROC。单次冻结推理（K=1），用 `1 − 最大类别概率` 排序，以四类 argmax 错误为正类。使用同一固定前景遮罩，平局取平均秩。全对或全错的帧记为缺失。对患者内有效 ED／ES 等权平均，再按患者等权报告均值、样本标准差和有效帧及患者数。

本地验收：

```bash
python -m acdc_benchmark smoke --output-dir /tmp/acdc-smoke
python -m pytest -q acdc_benchmark/tests
```
