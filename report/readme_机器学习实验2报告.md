# 机器学习实验：基于 Word2Vec 与稀疏文本模型的情感预测

## 1. 学生信息
- **姓名**：徐一恒
- **学号**：112304260142
- **班级**：数据1231

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于 Kaggle 竞赛 `Bag of Words Meets Bags of Popcorn`，围绕 IMDB 电影评论情感分类任务展开。实验前期以 **Word2Vec 词向量建模** 为主，完成文本向量化、传统分类与神经网络分类；在后续优化阶段，又进一步引入 **TF-IDF 稀疏特征、LinearSVC、NB-SVM 和交叉验证融合** 来提升线上成绩。

本实验重点包括：
- 文本预处理与清洗
- Word2Vec 训练与句向量表示
- 传统分类模型与深度学习模型比较
- 稀疏文本模型构建与融合
- Kaggle 提交、评分分析与结果优化

需要特别说明的是：**本比赛官方评价指标是 ROC-AUC，而不是 Accuracy**。因此本实验中所有模型比较与最终方案选择，均以 `ROC-AUC` 为核心依据。

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn（词语袋遇上爆米花袋）
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **最佳成绩对应提交日期**：2026-04-15

- **GitHub 仓库地址**：https://github.com/EHNGCCC/112304260142xuyiheng
- **GitHub README 地址**：https://github.com/EHNGCCC/112304260142xuyiheng/blob/main/README.md

本次实验中，我已经完成以下 GitHub 相关工作：
- 将 `formal` 相关代码、实验报告、结果文件整理为适合上传 GitHub 的仓库结构
- 使用 Git 对本地实验过程进行版本管理
- 配置本地电脑与 GitHub 的 SSH 连接
- 将项目成功推送到远程仓库，便于后续继续更新和回退历史版本

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
我在 Kaggle 上进行了多轮提交与优化，最终最高成绩如下：

- **最佳提交文件**：`submission_cv_sparse_rank_blend_auc.csv`
- **Public Score**：`0.97166`
- **Private Score**：`0.97166`
- **排名**：当前截图中未显示，故暂不填写

此外，其余几次代表性提交成绩如下：
- `submission_bilstm_auc.csv`：`0.95000`
- `submission_blend_auc.csv`：`0.95385`
- `submission_tfidf_svm_blend_auc.csv`：`0.96739`
- `submission_mega_rank_blend_auc.csv`：`0.97083`
- `submission_multiseed_cv_rank_blend_auc.csv`：`0.97157`
- `submission_pseudolabel_rank_blend_auc.csv`：`0.97137`

从结果可以看出，后期的 **稀疏文本交叉验证融合模型** 明显优于前期单纯依赖 Word2Vec 的方案。

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
我对影评文本做了较完整的清洗和标准化处理，既服务于 Word2Vec，也服务于后续的稀疏文本模型。主要步骤如下：
- 读取 `labeledTrainData.tsv`、`unlabeledTrainData.tsv` 和 `testData.tsv`
- 去除 HTML 标签和 `<br />` 换行标记
- 将文本统一转换为小写
- 展开部分英文否定缩写，例如 `don't -> do not`
- 去除大部分无效符号和非字母字符
- 按单词切分文本
- 去除英文停用词，但保留 `not`、`no`、`never` 等否定词

在后续实验中，我实际上维护了两种文本视图：
- `clean_review`：更标准、更彻底的清洗结果
- `soft_review`：保留更多原始 token 边界，便于稀疏特征建模

这样处理后，一方面可以提升 Word2Vec 训练质量，另一方面也有助于 `word n-gram` 和 `char n-gram` 模型提取有效特征。

---

### （2）Word2Vec 特征表示
在实验前期，我按照课程主题重点实现了 Word2Vec 路线。具体方法如下：
- 使用有标签训练集和无标签训练集共同训练 Word2Vec
- 词向量维度设置为 `200`
- 训练参数中使用窗口大小 `5`、最小词频 `3`
- 采用 **TF-IDF 加权平均词向量** 作为句子表示，而不是简单平均

除句向量外，我还补充了少量人工统计特征，例如：
- 文本长度
- 正负情感词比例
- 否定词比例
- 感叹号与问号数量

这些特征主要用于增强模型对评论情绪强度的表达能力。

需要说明的是：**Word2Vec 是本实验的重要基础路线，但不是最终线上得分最高的路线。** 它帮助我完成了课程主题要求，也为后续模型比较提供了可靠基线。

---

### （3）分类模型与最终方案
本实验不是一次成型，而是经过了多轮逐步优化。为了让实验过程更清楚，我将整个模型探索过程分为五个阶段。

#### 第一阶段：完成课程主题要求的 Word2Vec 基线
这一阶段的目标，是先按照“文本清洗 + Word2Vec + 分类器”的基本思路建立一个可运行的完整流程，确保项目从数据读取到 Kaggle 提交全部打通。

这一阶段主要完成了：
- 训练基于比赛语料的 `Word2Vec`
- 构建 `TF-IDF` 加权句向量
- 补充少量人工统计特征
- 训练传统分类模型与神经网络模型

尝试的模型包括：
- `Word2Vec + Logistic Regression`
- `Word2Vec + Random Forest`
- `Word2Vec + BiLSTM`

本地验证结果如下：
- `Logistic Regression`：`AUC = 0.9479`
- `Random Forest`：`AUC = 0.9238`
- `BiLSTM`：`AUC = 0.9581`

这一阶段的结论是：
- `BiLSTM` 在 Word2Vec 路线中效果最好
- `Random Forest` 在该任务上不如线性模型和神经网络
- Word2Vec 路线可以作为可靠基线，但仍有较大提升空间

对应 Kaggle 结果中，`submission_bilstm_auc.csv` 得到了 `0.95000`。

#### 第二阶段：在 Word2Vec 路线内部做融合优化
在第一阶段的基础上，我继续尝试将 Word2Vec 路线中的强模型进行融合，希望在不完全换路线的前提下继续提高线上成绩。

这一阶段的主要做法是：
- 保留前面表现较好的 `BiLSTM`
- 引入更强的传统模型作为补充
- 对不同模型输出的概率进行融合

这一阶段的代表性结果是：
- `submission_blend_auc.csv`：`0.95385`

这一阶段的结论是：
- 模型融合比单一 `BiLSTM` 略好
- 但整体提升仍然有限
- 说明仅在 Word2Vec 路线内部做微调，很难继续获得大幅增益

#### 第三阶段：转向更强的稀疏文本模型
在对比分析之后，我发现这个任务更适合使用稀疏文本特征，而不是单纯依赖稠密词向量。因此实验重点开始转向 `TF-IDF`、`n-gram` 和线性模型。

这一阶段主要引入了：
- `word TF-IDF + LinearSVC`
- `char TF-IDF + LinearSVC`
- `word n-gram`
- `char n-gram`

这一阶段的代表性提交为：
- `submission_tfidf_svm_blend_auc.csv`：`0.96739`

这一阶段的结论是：
- 稀疏文本特征明显优于 Word2Vec 句向量
- `word n-gram` 和 `char n-gram` 对情感词、否定结构和语气表达更加敏感
- 这一步确认了后续高分路线应该继续沿着稀疏模型展开

#### 第四阶段：交叉验证稀疏融合，得到当前最优结果
在确认稀疏文本路线更强之后，我继续将多个稀疏模型做系统融合，并引入交叉验证，使模型选择更加稳定、泛化更强。

这一阶段的主要做法包括：
- `word TF-IDF + LinearSVC`
- `char TF-IDF + LinearSVC`
- `word NB-SVM`
- `char NB-SVM`
- `5-fold cross-validation`
- `rank-average blending`

之所以使用 `rank blend`，是因为不同模型输出的概率分布不完全一致，而做排名融合往往能减少分布尺度不同带来的影响，提高集成稳定性。

这一阶段最终获得了全项目最佳成绩：
- **最佳提交文件**：`submission_cv_sparse_rank_blend_auc.csv`
- **Public Score**：`0.97166`
- **Private Score**：`0.97166`

这一阶段的结论是：
- 交叉验证比单次验证划分更稳定
- `LinearSVC` 与 `NB-SVM` 的互补性较好
- 稀疏特征 + 融合策略是当前项目里最有效的方案

#### 第五阶段：继续冲分的后续实验
在达到 `0.97166` 之后，我没有立刻停止，而是继续尝试一些更激进的优化方案，希望进一步冲高分数。

这一阶段测试了：
- `mega rank blend`
- `multi-seed cross-validation blend`
- `pseudo-labeling`

对应线上成绩包括：
- `submission_mega_rank_blend_auc.csv`：`0.97083`
- `submission_multiseed_cv_rank_blend_auc.csv`：`0.97157`
- `submission_pseudolabel_rank_blend_auc.csv`：`0.97137`

这一阶段的结论是：
- 多 seed 融合在本地验证中有轻微提升，但没有超过当前 best
- 伪标签方法对这个任务帮助不大，线上反而略低
- 本项目最终应保留 `submission_cv_sparse_rank_blend_auc.csv` 作为正式最优结果

因此，本实验最终采用的最佳方案是：  
**稀疏文本特征 + LinearSVC / NB-SVM + 交叉验证 + rank blend**

---

## 7. 实验流程
结合整个实验过程，我的完整流程如下：

1. 阅读 Kaggle 比赛要求，确认官方评价指标为 `ROC-AUC`
2. 参考老师提供的 `mould` 模板，在 `formal` 文件夹中搭建正式项目结构
3. 实现数据读取、文本清洗、Word2Vec 训练和句向量生成，完成第一版基线
4. 训练 `Logistic Regression`、`Random Forest` 和 `BiLSTM`，比较不同 Word2Vec 模型的 AUC
5. 发现 `BiLSTM` 是 Word2Vec 路线中最优方案，并完成第一批 Kaggle 提交
6. 在 Word2Vec 路线内部继续尝试模型融合，验证融合是否能进一步提高成绩
7. 为提高实验效率，加入预处理缓存、Word2Vec 缓存和特征缓存，减少重复计算时间
8. 转向更强的稀疏文本路线，构建 `TF-IDF`、`LinearSVC`、`NB-SVM` 等模型
9. 基于多种稀疏模型做交叉验证与 rank blending，得到当前最佳结果 `0.97166`
10. 在最佳方案基础上继续测试 `mega blend`、多 seed 融合和伪标签方法
11. 对各阶段线上分数进行比较，确认后续尝试未超过当前 best
12. 最终保留 `submission_cv_sparse_rank_blend_auc.csv` 作为正式结果
13. 将代码、实验报告、结果和修改总结整理为 GitHub 合规仓库，并完成本地 Git 与 GitHub 连接

---

## 8. 文件说明
本实验最终整理了一个适合上传 GitHub 的项目仓库，主要结构如下：

```text
112304260142xuyiheng/
├─ code/
│  ├─ main.py
│  ├─ generate_highscore_submission.py
│  ├─ generate_multiseed_cv_submission.py
│  ├─ generate_pseudolabel_submission.py
│  ├─ requirements.txt
│  └─ src/
│     ├─ preprocess.py
│     ├─ feature_engineering.py
│     ├─ classical_pipeline.py
│     ├─ bilstm_pipeline.py
│     ├─ ensemble_pipeline.py
│     ├─ cache_utils.py
│     └─ 其他辅助模块
├─ report/
│  ├─ experiment_report.md
│  ├─ modification_summary.md
│  └─ github_setup_guide.md
├─ results/
│  ├─ metrics/
│  ├─ plots/
│  ├─ submissions/
│  └─ score_summary.md
├─ .gitignore
└─ README.md
```

各部分作用如下：
- `code/`：存放项目代码，包括主入口、各类模型脚本和核心模块
- `code/src/`：存放预处理、特征工程、模型训练与融合逻辑
- `report/`：存放实验报告、修改内容总结和 GitHub 使用说明
- `results/metrics/`：存放各阶段实验的 AUC、融合权重等统计结果
- `results/plots/`：存放训练曲线和模型比较图
- `results/submissions/`：存放导出的 Kaggle 提交文件
- `results/score_summary.md`：汇总各次提交的线上分数
- `README.md`：仓库首页说明
- `.gitignore`：排除大数据文件、缓存和模型中间产物，避免仓库过大

如果从本地开发角度看，`kaggle1/` 目录下还保留了：
- `mould/`：老师提供的参考模板
- `word2vec-nlp-tutorial/`：比赛原始数据
- `formal/`：本地实验主目录
- `formal_github_repo/`：整理后用于上传 GitHub 的仓库目录

---

## 9. 修改内容总结
根据整个实验过程，本项目的主要修改和优化包括：

1. 从老师模板出发，建立了完整的 `formal` 项目结构
2. 将评价指标从容易误用的 Accuracy 明确改为官方要求的 `ROC-AUC`
3. 实现了 Word2Vec、经典分类器和 BiLSTM 基线
4. 增加了预处理缓存、Word2Vec 缓存和特征缓存，加快实验速度
5. 引入更强的稀疏文本模型，包括 `TF-IDF`、`LinearSVC` 和 `NB-SVM`
6. 实现交叉验证、rank blend、多 seed 融合和伪标签等优化方案
7. 生成多份 Kaggle 提交文件并持续比较线上成绩
8. 整理出适合 GitHub 上传的 `code / report / results` 结构
9. 完成本地 Git 配置、SSH 密钥配置以及远程仓库推送

---

## 10. 实验总结
本实验最初围绕 Word2Vec 展开，完成了课程主题要求的词向量建模流程；但在实际 Kaggle 优化过程中，我发现对于 IMDB 情感分类任务，**稀疏文本特征与线性模型的效果更优**。最终最好的线上成绩不是来自 Word2Vec 平均向量路线，而是来自交叉验证下的稀疏文本融合模型。

通过这次实验，我不仅完成了文本分类建模，也完成了更完整的工程化实践，包括：
- 多轮实验与对比分析
- 提交文件管理
- GitHub 仓库整理与版本控制
- 本地与远程仓库连接

最终结果表明：
- 比赛指标必须严格以 `ROC-AUC` 为准
- Word2Vec 适合作为课程主题和建模基础
- 稀疏特征在该任务上更有优势
- 交叉验证与模型融合对最终成绩提升最明显
