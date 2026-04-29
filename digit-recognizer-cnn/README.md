# CNN 手写数字识别实验项目

本项目完成 Kaggle Digit Recognizer 手写数字识别实验，包含 CNN 对比实验、最终提交模型、预测提交文件和 Flask Web 应用。

## 项目结构

```text
digit-recognizer/
├── app/                         # Flask Web 应用
├── data/                        # Kaggle 原始数据
├── models/                      # 训练好的模型权重
├── outputs/
│   ├── history/                 # 训练历史
│   ├── plots/                   # Loss 曲线
│   ├── submissions/             # Kaggle 提交文件
│   └── tables/                  # 实验结果表
├── report/                      # 完成版实验报告
├── src/                         # 训练、实验和生成提交脚本
├── render.yaml                   # Render 部署配置
├── requirements.txt              # Web 部署最小依赖
├── requirements-dev.txt          # 本地训练/实验依赖
└── README.md
```

## 运行方式

安装本地训练依赖：

```powershell
& 'E:\machining learning\.venv\Scripts\python.exe' -m pip install -r requirements-dev.txt
```

运行 4 组对比实验：

```powershell
& 'E:\machining learning\.venv\Scripts\python.exe' .\src\run_experiments.py
```

使用最终模型重新生成 Kaggle 提交：

```powershell
& 'E:\machining learning\.venv\Scripts\python.exe' .\src\generate_submission.py
```

启动 Flask Web 应用：

```powershell
& 'E:\machining learning\.venv\Scripts\python.exe' .\app\app.py
```

默认访问地址为 `http://127.0.0.1:7860`。

如果提示端口被占用，可以先查看并停止旧的 Python 服务：

```powershell
netstat -ano | Select-String ':7860'
Stop-Process -Id <PID> -Force
```

## Render 部署

由于 GitHub 仓库根目录同时包含多个项目，推荐在 Render 中手动创建 Web Service，并把 Root Directory 设置为 `digit-recognizer-cnn`。

推荐设置：

```text
Service Type: Web Service
Root Directory: digit-recognizer-cnn
Build Command: pip install -r requirements.txt
Start Command: gunicorn --chdir app app:app
Python Version: 3.12.10
```

部署说明：

- GitHub 仓库中保留 `models/final_cnn.pt`，用于 Web 应用在线预测。
- Kaggle 原始 `train.csv` / `test.csv` 不上传 GitHub，需要本地训练时自行放入 `data/`。
- Render 上只运行 Web 应用，不重新训练模型；依赖文件使用 CPU 版 PyTorch，适合 Render 免费实例部署。

## 关键结果

- 最终 Kaggle Score：`0.99564`
- 最终模型：`models/final_cnn.pt`
- Kaggle 提交文件：`outputs/submissions/submission_full_ensemble.csv`
- 对比实验结果：`outputs/tables/comparison_results.csv`
- 对比实验 Loss 曲线：`outputs/plots/comparison_loss_curves.png`
