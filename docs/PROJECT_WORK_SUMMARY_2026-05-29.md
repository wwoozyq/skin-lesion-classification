# 项目工作梳理报告

**生成日期**：2026-05-29
**用途**：交付前对全部已完成工作、Git 历史、实验结果、成员分工的总盘点。
**仓库路径**：`/Users/wlm/Desktop/HW/skin-lesion-classification`
**远端**：`origin/main`（已与本地同步至 `28ae12d`）

---

## 1. Git 提交时间轴（全部 39 次提交，按时间倒序）

**5 人小组**，git 上有 4 位贡献者，第 5 位无 GitHub 账号，其工作由王礼铭代为提交：

| 真实姓名 | GitHub 账号 | 提交次数 | 主要贡献 |
|---|---|---:|---|
| **王礼铭** | `wwoozyq` | 35 | 主要开发者：主模型、级联、融合、文档、整合 |
| **刘智瑞** | `liuliuzhizhi` | 3 | 深度学习扩展（MobileNetV2）、Kaggle 指南、团队进度文档 |
| **章蓉** | `rongz2227-ux` | 1 | **ABCD 特征 + SVM-RBF + melanoma 阈值调整**（grouped split）：Macro-F1=0.7227、Balanced Accuracy=0.7145、mel recall=0.6857；分支 `origin/abcd-grouped-optimization`，提交 `9ce3cae`（2026-05-19 12:27）|
| **颜熙** | `saltyout` | 1 | 待补：具体贡献内容 |
| **王昱航** | 无（由 `wwoozyq` 代提交）| — | **边界/形状特征增强**：`src/features_boundary.py`（339 行），由王礼铭于 2026-05-17 16:16 通过 `f321c31` 代为提交，含 `boundary_integration_results.csv` 实验结果 |

### 1.1 时间轴（按阶段分）

| 提交哈希 | 日期时间 | 阶段 | 内容 |
|---|---|---|---|
| `dfcd083` | 2026-05-06 16:24 | **阶段 0：起步** | 项目骨架与基线 |
| `55f1963` | 2026-05-06 17:20 | 阶段 0 | data 软链 .gitignore |
| `1615c30` | 2026-05-07 23:41 | **阶段 1：基础特征** | lesion-background 对比度特征 |
| `bc46e70` | 2026-05-07 23:45 | 阶段 1 | 优化提交指南文档 |
| `05f7733` | 2026-05-08 13:43 | 阶段 1 | ABCD-inspired 特征首版 |
| `adb3d3f` | 2026-05-08 15:22 | 阶段 1 | 增强版 ABCD-inspired 特征 |
| `78a4493` | 2026-05-08 20:12 | 阶段 1 | 外部数据集调研文档 |
| `bfc1426` | 2026-05-08 21:26 | 阶段 1 | 伪 mask 预处理脚本 |
| `bca00dc` | 2026-05-08 21:34 | 阶段 1 | 伪 mask 选择更保守 |
| `1bc329d` | 2026-05-08 21:38 | 阶段 1 | lesion-centered crop 预处理 |
| `3786424` | 2026-05-08 23:02 | 阶段 1 | external crop 实验流程 |
| `0850ef8` | 2026-05-09 16:13 | 阶段 1 | 修：本地代理用于 ISIC 下载 |
| `f321c31` | 2026-05-17 16:16 | **阶段 2：边界与纹理** | boundary 特征整合到 ABCD pipeline |
| `0f16040` | 2026-05-17 12:16 | 阶段 2 | LBP/GLCM/gradient 纹理特征 + SVM 分类器 |
| `3906fba` | 2026-05-18 18:56 | 阶段 2 | 文档：grouped validation 实验 |
| `962c054` | 2026-05-19 11:10 | **阶段 3：分组验证修正** | Kaggle 深度学习指南 |
| `7cebf4c` | 2026-05-19 11:29 | 阶段 3 | 文档：团队进度总览 |
| `9ce3cae` | 2026-05-19 12:27 | 阶段 3 | 优化 ABCD（grouped 验证）|
| `f4618ef` | 2026-05-19 18:22 | 阶段 3 | **整合 grouped ABCD（主模型定型）** |
| `3517457` | 2026-05-20 00:25 | **阶段 4：深度学习扩展** | 低负载深度夜跑器 |
| `0f3d0c5` | 2026-05-20 00:29 | 阶段 4 | 可配置低负载深度夜跑器 |
| `0261c19` | 2026-05-20 00:29 | 阶段 4 | 同上（合并冲突）|
| `74d7c14` | 2026-05-20 00:32 | 阶段 4 | 合并远端深度跑器更新 |
| `ea37068` | 2026-05-20 10:03 | 阶段 4 | 修：深度准确率阈值仅作参考 |
| `1d15fb3` | 2026-05-20 13:58 | 阶段 4 | 文档：深度学习扩展结果 |
| `fc88b11` | 2026-05-21 10:26 | **阶段 5：探索性方法** | 初次提交（队友推送）|
| `9be076a` | 2026-05-21 11:16 | 阶段 5 | 图与结果（队友推送）|
| `5603340` | 2026-05-21 11:19 | 阶段 5 | 模型（队友推送）|
| `88544be` | 2026-05-21 15:58 | 阶段 5 | 文档：组织实验证据与验证 |
| `e8b2270` | 2026-05-28 20:23 | **阶段 6：最后冲刺** | 实验：TTA + 嵌套 CV 融合权重（负结果）|
| `3a5e7a4` | 2026-05-28 20:52 | 阶段 6 | 文档：TTA + 嵌套 CV 融合负结果记录到账本 |
| `4304a5b` | 2026-05-28 22:47 | 阶段 6 | 文档：lesion-hard 错误分析（解释 0.78 上限）|
| `4990471` | 2026-05-28 23:10 | 阶段 6 | 实验：去毛发预处理（负结果）|
| `0ef5366` | 2026-05-28 23:11 | 阶段 6 | 文档：去毛发预处理负结果记录 |
| `e1a59ed` | 2026-05-28 23:21 | 阶段 6 | 文档：完整探索时间线 HTML（用于 PPT）|
| `84b84be` | 2026-05-29 00:10 | 阶段 6 | 文档：6/10 提交前突破天花板的研究计划 |
| `ddae438` | 2026-05-29 01:11 | 阶段 6 | 实验：医学预处理 × 级联探索（夜跑）|
| `abac3c0` | 2026-05-29 01:16 | 阶段 6 | 修：夜跑 pass/fail 锚定 cell 0 而非账本 0.7887 |
| `f2b29b7` | 2026-05-29 11:47 | 阶段 6 | 实验：皮肤镜结构特征 + 级联变体扫描工具 |
| `28ae12d` | 2026-05-29 11:47 | 阶段 6 | 文档：皮肤镜结构特征负结果 + 夜跑文档重写 |

### 1.2 分支总览

| 分支 | 用途 | 是否合并 |
|---|---|---|
| `main` | 主干（默认）| — |
| `model` | 早期级联模型分支 | 已合并入 main |
| `tta-nested-fusion` | TTA + 嵌套 CV 融合（负结果）| **未合并**，作为负结果保留 |
| `hair-removal-eval` | 去毛发预处理（负结果）| **未合并**，作为负结果保留 |
| `feature/abcd`, `feature/abcd-clean` | ABCD 特征探索 | 已并入 main |
| `feature/boundary-integration` | 边界特征整合 | 已并入 main |
| `feature/contrast-example` | 对比度特征示例 | 已并入 main |
| `feature/external-data-research` | 外部数据集调研 | 已并入 main（仅文档，未用外部数据）|
| `feature/texture` | 纹理特征 | 已并入 main |
| `abcd-grouped-optimization` | ABCD grouped 优化（主模型来源）| 已并入 main |
| `codex/integrate-abcd-grouped` | ABCD grouped 整合 | 已并入 main |
| `codex/traditional-ml-score-search` | 传统 ML 分数搜索 | 已并入 main |

---

## 2. 当前已完成工作分类总览

### 2.1 数据层

| 模块 | 文件 | 状态 |
|---|---|---|
| 数据读取 | `src/dataset.py` | ✅ |
| `base_id` 工具（防泄漏关键）| `src/utils.py::base_id_from_image_id` | ✅ |
| 数据完整性检查 | `scripts/check_dataset.py` | ✅ |
| Grouped CV 输出验证 | `scripts/check_grouped_cv_outputs.py` | ✅ |

### 2.2 特征层（共 12 个特征模块）

| 特征模块 | 文件 | 是否被主模型采用 |
|---|---|---|
| 颜色（RGB/HSV/Lab）| `src/features_color.py` | ✅（间接，通过 `all_abcd_grouped`）|
| 形状 | `src/features_shape.py` | ✅ |
| 纹理（LBP/GLCM/Sobel）| `src/features_texture.py` | ✅ |
| Lesion-background 对比度 | `src/features_contrast.py` | ✅ |
| ABCD-inspired v2 | `src/features_abcd.py` | 历史版本 |
| **ABCD grouped（主特征集）**| **`src/features_abcd_grouped.py`** | ✅ **主模型核心** |
| 边界（凸包/分形）| `src/features_boundary.py` | ✅ |
| Mel/NV 错误驱动 | `src/features_melnv.py` | ❌ 负结果 |
| 多尺度 LBP | `src/features_lbp_multi.py` | ❌ 未采用 |
| Gabor 纹理 | `src/features_gabor.py` | ❌ 未采用 |
| 子区域不对称 | `src/features_subregion.py` | ❌ 未采用 |
| 皮肤镜结构（22 维）| `src/features_dermoscopy.py` | ❌ **5 月 29 日确认负结果** |
| 统一入口 | `src/features.py` | ✅ |

### 2.3 训练与预测层

| 模块 | 文件 | 状态 |
|---|---|---|
| 传统 ML 主训练 | `src/train_ml.py` | ✅ |
| XGBoost 级联训练 | `src/train_ml_cascade.py` | ✅ |
| 主模型 + 级联融合训练 | `src/train_ml_fusion.py` | ✅ |
| 医学预处理（color constancy 等）| `src/preprocess_medical.py` | ✅（负结果保留）|
| 最终 `output.csv` 生成 | `run.py` | ✅ 支持 normal/cascade/fusion 三种 bundle |

### 2.4 实验脚本（22 个 + 多个最近新增）

| 脚本 | 用途 | 关联 ledger 节 |
|---|---|---|
| `experiments/run_ml_grid.py` | 网格搜索 | §2 |
| `experiments/run_stability.py` | 多种子稳定性 | §1 |
| `experiments/run_abcd_grouped_integration.py` | ABCD grouped 整合 | §8 |
| `experiments/run_melnv_refinement.py` | Mel/NV 二阶段精炼（负）| §7 |
| `experiments/analyze_robustness.py` | 鲁棒性分析 | §4 |
| `experiments/visualize_errors.py` | 错误可视化 | §5 |
| `experiments/run_xgb_cascade_search.py` | XGBoost 级联搜索 | §9 |
| `experiments/run_early_fusion_search.py` | 前端特征融合 | §10 |
| `experiments/run_fusion_ensemble.py` | 后端概率融合 | §11 |
| `experiments/train_deep.py` | 深度学习扩展 | §14 |
| `experiments/run_tta_nested_fusion.py` | TTA + 嵌套 CV（负）| §12 |
| `experiments/run_hair_removal_eval.py` | 去毛发预处理（负）| §13 |
| `experiments/run_overnight_exploration.py` | 医学预处理夜跑 | §15 |
| `experiments/run_variant_sweep_a1.py` | 18 配置变体扫描 | §15 |
| `experiments/run_dermoscopy_features.py` | 皮肤镜结构特征（负）| §16 |
| `experiments/compare_lr_cascade_errors.py` | LR vs 级联 错误重叠分析 | 5/29 新增 |
| `experiments/verify_per_class_fusion.py` | 逐类融合 Path A（负）| 5/29 新增 |
| `experiments/verify_structural_fusion.py` | 结构 2 轴融合 Path A'（负）| 5/29 新增 |

### 2.5 文档层（共 24 份）

| 文档 | 用途 |
|---|---|
| `README.md` | 项目入口 |
| `docs/PROJECT_REVIEW_GUIDE.md` | 给评审看的项目主线 |
| `docs/EXPERIMENT_LEDGER.md` | **完整实验账本（16 节）** |
| `docs/REPRODUCIBILITY_CHECKLIST.md` | 复现命令清单 |
| `docs/results_summary.csv` | 紧凑结果表（PPT 用）|
| `docs/STRICT_GROUPED_RESULTS.md` | 主模型严格 grouped CV 结果 |
| `docs/ABCD_GROUPED_INTEGRATION.md` | ABCD grouped 整合细节 |
| `docs/XGB_CASCADE_INTEGRATION.md` | 级联整合细节 |
| `docs/EARLY_FUSION_RESULTS.md` | 前端融合结果 |
| `docs/FUSION_ENSEMBLE_RESULTS.md` | 后端融合结果 |
| `docs/DEEP_LEARNING_EXTENSION_RESULTS.md` | 深度学习扩展结果 |
| `docs/TEXTURE_OPTIMIZATION.md` | 纹理优化 |
| `docs/LESION_HARD_ERROR_ANALYSIS.md` | "硬病例"错误分析 |
| `docs/OVERNIGHT_EXPLORATION_PLAN.md` | 夜跑计划 |
| `docs/OVERNIGHT_EXPLORATION_RESULTS.md` | 夜跑结果（负）|
| `docs/DERMOSCOPY_FEATURES_RESULTS.md` | 皮肤镜结构特征（负）|
| `docs/SUBMISSION_RESEARCH_PLAN.md` | 提交前研究计划 |
| `docs/EXPLORATION_TIMELINE.html` | 完整探索时间线（PPT 用）|
| `docs/PRESENTATION_HIGHLIGHTS.md` | 答辩亮点 |
| `docs/TEAM_PROGRESS_OVERVIEW.md` | 团队进度总览 |
| `docs/TEAM_TASKS.md` | 团队分工 |
| `docs/CURRENT_EXPERIMENT_PROGRESS.md` | 当前实验进度 |
| `docs/EXPERIMENT_PLAN.md` | 实验计划 |
| `docs/FULL_SCORE_EXPERIMENT_PLAN.md` | 满分计划 |
| `docs/KAGGLE_DEEP_LEARNING.md` | Kaggle 深度学习指南 |

---

## 3. 全部实验结果（按账本顺序）

> 评估协议（除非明确标注）：`StratifiedGroupKFold`，按 `base_id` 分组，5 折，5 种子（42 / 127 / 2024 / 3407 / 520）。

### 3.1 主线实验结果总表

| # | 实验 | 协议 | Acc | Macro-F1 | BalAcc | 角色 | 文件 |
|---|---|---|---:|---:|---:|---|---|
| 1 | Earlier boundary LR | seed 127 grouped OOF | 0.7233 | 0.7454 | 0.7535 | 历史 baseline | `experiments/run_ml_grid.py` |
| 1b | **章蓉：ABCD + SVM-RBF + mel 阈值调整** | seed 127 grouped OOF | — | **0.7227** | **0.7145** | 阈值调整 ablation（mel recall 0.6857）| 分支 `origin/abcd-grouped-optimization` (`9ce3cae`) |
| 2 | **主 LR**（`all_abcd_grouped + LR03 + k=140`）| seed 127 grouped OOF | **0.7600** | **0.7715** | **0.7871** | **稳定主模型** | `src/train_ml.py` |
| 3 | 主 LR | 5 种子 mean | 0.7283 | 0.7435 | **0.7512** | 稳定性估计 | 同上 |
| 4 | 主 LR Macro-F1 std | 5 种子 std | — | 0.0271 | — | 稳定性 | 同上 |
| 5 | XGBoost 级联（早期 smoke）| seed 127 grouped OOF | 0.7617 | 0.7802 | 0.8017 | 探索高分 | `experiments/run_xgb_cascade_search.py` |
| 6 | XGBoost 级联（早期）| 5 种子 bagged best | 0.7500 | 0.7655 | **0.7887** | 探索稳定性 | 同上 |
| 7 | XGBoost 级联（`deeper, k=120, no-pp`，5/29 sweep 最强）| 5 种子 bagged | 0.7450 | 0.7609 | **0.7839** | 当前级联候选 | `experiments/run_variant_sweep_a1.py` |
| 8 | Early fusion（特征级）| seed 127 grouped OOF | 0.7700 | 0.7830 | 0.7997 | 前端融合 ablation | `experiments/run_early_fusion_search.py` |
| 9 | Early fusion | 5 种子 mean best | 0.7450 | 0.7618 | 0.7766 | 前端融合稳定性 | 同上 |
| 10 | Late fusion（0.5 LR + 0.5 cascade）| seed 127 grouped OOF | 0.7717 | 0.7909 | **0.8055** | 单种子最高 | `experiments/run_fusion_ensemble.py` |
| 11 | Late fusion | 5 种子 bagged | 0.7417 | 0.7607 | **0.7763** | 后端融合稳定性 | 同上 |
| 12 | MobileNetV2（深度学习扩展）| grouped validation | **0.8750** | **0.8623** | **0.8533** | **深度扩展，不进定量提交** | `experiments/train_deep.py` |

### 3.2 负结果详表（很重要，能讲）

| # | 负实验 | 5 种子 bagged BalAcc | Δ vs 强级联 0.7839 | 负结果原因 | 出处 |
|---|---|---:|---:|---|---|
| §3 | Mask cleaning | — | — | 原始 mask 已经够好 | `docs/EXPERIMENT_LEDGER.md` §3 |
| §6 | Mel/NV 错误驱动特征 | — | — | 一致性不足 | §6 |
| §7 | Mel/NV 二阶段精炼 | — | — | 阈值漂移 | §7 |
| §12 | TTA + 嵌套 CV 融合 | 0.7726（cell D）| -0.0113 | bagging 已吸收 augmentation 方差 | §12 |
| §13 | 去毛发预处理（DullRazor 风）| — | 负 | 只 2 张毛发遮挡，杠杆太小 | §13 |
| §15-A1 | shades_of_gray 预处理（强级联上）| 0.7821 | -0.0018 | 强级联已正则化，多余 | §15 |
| §15-A2 | CLAHE 预处理 | 0.7468–0.7747 | -0.01~-0.04 | 损坏色度信号 | §15 |
| §15-A3 | hb_melanin 色素分解 | 0.7466 | -0.04 | 同上 | §15 |
| §15-B1 | Stage 2 加 abcd_grouped | 0.7676 | -0.0163 | 与 A1 冗余 | §15 |
| §16 | 皮肤镜结构特征（22 维）| 0.7722 | -0.0117 | 与 ABCD 重复，asymmetry 在级联反向作用 | §16 |
| 5/29 | Per-class 融合（Path A）| seed 127 nested = 0.7910 | -0.0145（vs 平均融合）| w_mel std=0.258 不稳定 | `experiments/verify_per_class_fusion.py` |
| 5/29 | 结构 2 轴融合（Path A'）| seed 127 nested = 0.7907 | -0.0095（vs 平均融合）| w2 std=0.244 不稳定 | `experiments/verify_structural_fusion.py` |

### 3.3 鲁棒性 / 错误分析数据点

| 指标 | 数值 | 出处 |
|---|---:|---|
| 原图 vs 增强图 准确率（主 LR）| 0.7800 vs 0.7500 | `experiments/analyze_robustness.py` |
| Group 一致性 | 0.7350 | 同上 |
| LR 主模型错误总数（seed 127）| 144 / 600 | `compare_lr_cascade_errors.py` |
| 级联错误总数（seed 127）| 146 / 600 | 同上 |
| LR + 级联 共同错误 | 89 | 同上 |
| 仅 LR 错（级联对）| 55 | 同上 |
| 仅级联错（LR 对）| 57 | 同上 |
| 错误重叠率（占 LR 错的比例）| 62% | 同上 |
| LR 主模型混淆矩阵主要混淆 | mel ↔ nv | `experiments/visualize_errors.py` |

### 3.4 5/29 当天新增的融合分析（负结果）

| 实验 | seed 127 BalAcc | 5 种子 bagged | 结论 |
|---|---:|---:|---|
| Flat 0.5/0.5 融合（基线）| 0.8002 | — | 未做 5 种子 |
| **Structural 0.5/0.5 融合**（结构 2 轴）| **0.8039** | — | 比 flat 多对 1 个，未做 5 种子 |
| Path A 嵌套 CV（逐类权重）| 0.7910 | — | nested CV 学权重 → 回归 |
| Path A' 嵌套 CV（结构 2 轴权重）| 0.7907 | — | 同上回归 |
| Grid 上界（in-sample 过拟合上界，参考用）| 0.8106 | — | 仅说明权重学习的上限 |

**5/29 融合分析结论**：所有"让数据自己学融合权重"的方案都比固定 0.5/0.5 回归。n=600 + vasc 仅 90 张，权重选择方差太大。

---

## 4. 6/10 提交决策（需要你拍板）

### 4.1 三个候选

| 候选 | 5 种子 bagged BalAcc | seed 127 BalAcc | 风险 | 推荐度 |
|---|---:|---:|---|---|
| **A. 主 LR**（`all_abcd_grouped + LR03 + k=140`）| 0.7512 | 0.7871 | 最低，文档全锁定 | 安全 |
| **B. XGBoost 级联**（`deeper k=120 soft, no-pp`）| **0.7839** | 0.7941 | 中（per-seed std 0.0206），需要 5 种子 bagged 部署 | **建议** |
| C. Late fusion（0.5/0.5）| 0.7763 | 0.8055（单种子高分但虚）| 中 | 不推荐做主提交 |

### 4.2 当前部署状态

| 组件 | 状态 |
|---|---|
| `run.py` 三种 bundle 模式（normal/cascade/fusion）| ✅ 已就绪 |
| `outputs/models/ml_baseline.joblib`（主 LR）| ✅ 已存在 |
| `outputs/models/xgb_cascade_deeper_k120_soft.joblib`（级联单种子）| ✅ 已存在 |
| `outputs/models/ml_fusion_candidate.joblib`（融合）| ✅ 已存在 |
| **5 种子 bagged cascade trainer** | ❌ **待开发** |
| **`run.py cascade_bagged` 模式** | ❌ **待开发** |
| **最终提交压缩包 `Project-2_王礼铭.zip`** | ❌ **待打包** |

### 4.3 6/10 之前的待办（按候选 B 走）

```text
1. 写 src/train_ml_cascade_bagged.py：在全量 600 张上跑 5 种子 × 5 折 = 25 个 fold-model
2. 给 run.py 加 cascade_bagged 模式，预测时平均 25 个模型的 softmax
3. 跑一次 output.csv 验证流程（local data 上跑）
4. 写最终 PPT 和报告口径变更（从"主 LR"改为"主级联，LR 作为基线"）
5. 打包 Project-2_王礼铭.zip：源码 + run.py + outputs/models/cascade_bagged.joblib + README
```

---

## 5. 成员分工（5 人组）

### 5.1 基于 git 历史和实际工作的分工映射

| 成员 | 真实姓名 | GitHub 账号 | 实际承担模块 | 关键文件 / 产出 | 答辩可讲点 |
|---|---|---|---|---|---|
| **A. ABCD 特征 + SVM-RBF + 阈值调整** | **章蓉** | `rongz2227-ux` | **ABCD 特征探索 + SVM-RBF + melanoma 阈值调整** | 分支 `origin/abcd-grouped-optimization`（提交 `9ce3cae`）；阈值实验输出 `outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_melthr045_*`；SVM 输出 `outputs/metrics/ml_all_abcd_v2_svm_grouped_seed127_*` | 实验结果：**Macro-F1=0.7227、BalAcc=0.7145、mel recall=0.6857**；为什么阈值调整能提 mel recall 但牺牲 nv precision；SVM-RBF vs LR 的对比逻辑 |
| **B. 边界 / 形状特征增强** | **王昱航** | 无（由王礼铭代提交 `f321c31`，2026-05-17 16:16）| **边界形状特征增强** | `src/features_boundary.py`（339 行）, `docs/results/boundary_integration_results.csv` | 边界凸包 / 分形维度 / 边界复杂度的医学直觉、为什么边界特征对形状不规则病例（特别是 mel）有判别力 |
| **C. 数据协议 + 基础特征**（建议分给颜熙）| **颜熙** | `saltyout` | 待补：grouped CV 防泄漏 + color/shape/texture/contrast 基础特征工程 | `src/utils.py`, `scripts/check_grouped_cv_outputs.py`, `src/features_color.py`, `src/features_shape.py`, `src/features_texture.py`, `src/features_contrast.py` | 90% → 70% 的方法学修正、`base_id` 防泄漏、基础特征的医学直觉 |
| **D. 主模型 + 模型搜索 + 融合实验** | **王礼铭** | `wwoozyq` | ABCD grouped 整合、主分类器、网格搜索、级联、early/late fusion、负结果分析、6/10 提交统筹 | `src/features_abcd_grouped.py`, `src/train_ml.py`, `src/train_ml_cascade.py`, `experiments/run_*` 全部 22 个脚本, `docs/EXPERIMENT_LEDGER.md` | 主模型为什么选 LR + k=140；级联结构红利（bagged 0.7839）；融合三轮全测过都是负结果；6/10 提交决策；负结果"我们尝试了所有方法"叙事 |
| **E. 鲁棒性 + 错误分析 + 深度学习扩展** | **刘智瑞** | `liuliuzhizhi`（3 commits）| 深度学习扩展（MobileNetV2）、Kaggle 指南、团队进度文档 | `experiments/train_deep.py`, `experiments/run_deep_lowload_night.py`, `docs/KAGGLE_DEEP_LEARNING.md`, `docs/DEEP_LEARNING_EXTENSION_RESULTS.md`, `docs/TEAM_PROGRESS_OVERVIEW.md` | MobileNetV2 BalAcc 0.8533 vs 传统 0.7512；深度学习作为对照而非主线；为什么课程定量评估只能用传统 ML |

> ⚠️ 颜熙的具体贡献内容还没确认。如果他实际上做的是别的（比如错误分析、可视化、文档），把 C 这一行的内容替换掉即可。

### 5.2 验收任务（每人至少做一项）

| 成员 | 验收命令或材料 |
|---|---|
| A | 跑 `scripts/check_grouped_cv_outputs.py`，记录 PASS 输出（截图）|
| B | 列出 8-10 个基础特征例子并解释医学/图像直觉 |
| C | 复现或解释 `all_abcd_grouped + LR03` 的 seed 127 指标（0.7600 / 0.7715 / 0.7871）|
| D | 用 `docs/results_summary.csv` 做最终模型对比表 |
| E | 准备 robustness 表格 + mel/nv 错误样例图 |

### 5.3 每人必须能回答的问题

```text
1. 我负责的模块解决了什么问题？
2. 这个模块符合课程"传统方法"要求吗？
3. 使用了哪些输入和输出？
4. 对应实验结果是多少？
5. 最终是否被主模型采用？如果没有，为什么仍然有价值？
```

---

## 6. 课程截止时间表

| 截止 | 日期 | 内容 | 当前状态 |
|---|---|---|---|
| 代码包 | **2026-06-10** | `Project-2_王礼铭.zip`（源码 + 模型 + README）| ❌ 待打包 |
| PPT | 2026-06-16 | `Project-2_王礼铭.pptx` | 部分材料就绪（`docs/PRESENTATION_HIGHLIGHTS.md`、`docs/EXPLORATION_TIMELINE.html`），未拼成 PPT |
| 报告 | ~2026-07-06 | `Project-2_王礼铭.pdf` | 草稿未开始 |

距离 6/10 还有 **12 天**。

---

## 7. 当前最重要的三件事

1. **拍板候选 B 还是 A**：候选 B（级联）bagged 比 A 高 +0.033，证据扎实，但要改报告口径。
2. **写 5 种子 bagged 训练器**：这是 B 候选必须的部署活儿，~2 小时工作量。
3. **打包提交 zip**：6/10 截止前完成，含 `output.csv` 端到端跑通验证。

剩下的"提分尝试"全部是负结果（§12 / §13 / §15 / §16 / Path A / Path A'），不必再深挖。报告主线清晰：**严格 grouped CV + 可解释主线 + 利用结构先验 + 全方位负结果记录**。

---

## 8. 附录：本报告涉及的关键产物路径

```text
源码：
  src/                                 全部模块
  experiments/                         全部实验脚本
  scripts/check_grouped_cv_outputs.py  防泄漏验收

文档：
  docs/EXPERIMENT_LEDGER.md            全部实验账本（16 节）
  docs/PROJECT_REVIEW_GUIDE.md         给评审的导览
  docs/results_summary.csv             紧凑结果表

模型与缓存：
  outputs/models/                      已训练模型
  outputs/cache/                       特征缓存
  outputs/metrics/                     全部指标 CSV
  outputs/metrics/cascade_seed127_oof_predictions.csv     LR vs 级联 错误重叠分析输出
  outputs/metrics/structural_fusion_seed127.log           5/29 Path A' 跑出来的日志

待生成：
  outputs/models/cascade_bagged.joblib  ❌ 5 种子 bagged 部署模型
  output.csv                            ❌ 提交用的预测文件
  Project-2_王礼铭.zip                  ❌ 最终交付包
```
