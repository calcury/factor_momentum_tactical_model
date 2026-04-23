
# factor_momentum_tactical_model

本仓库实现了基于因子与动量的研究与回测流水线：从原始指数成分收益做特征/聚类分析、滚动窗口预测，再基于预测结果进行策略决策与回测对比。

## 项目结构

```
.
├─ main.py    # 运行完整流水线
├─ requirements.txt    # 依赖项
├─ README.md
├─ config/
│   └─ prediction_model_config.json
├─ data/
│   ├─ sp100.csv
│   ├─ hsi.csv
│   └─ stock_static.csv    # Task1 生成的静态特征表
├─ result/    # 运行结果
│   ├─ task1/
│   ├─ task2/
│   └─ task3/
└─ src/
   ├─ task1/
   │   ├─ feature_analysis.py    # 特征分析与静态表生成
   │   ├─ correlation_coefficient.py    # 相关性分析
   │   ├─ cluster_kmeans.py    # K-means 聚类
   │   └─ cluster_agglomerative.py    # 凝聚层次聚类
   ├─ task2/
   │   ├─ average_return.py    # 均值收益率分析
   │   ├─ variance_interpret.py    # PCA方差解释率计算
   │   ├─ prediction_model.py    # 滚动预测
   │   └─ evaluate.py    # 皮尔逊相关系数分析
   └─ task3/
	   ├─ decision_model.py    # 基于预测结果的决策与回测
	   └─ backtesting.py    # 完整回测流程
```

## 简要说明

- 目的：从成分股收益中提取因子与动量信号，使用滚动窗口训练并预测指标，最后构建决策引擎进行回测与对比分析
- 入口：运行根目录的 `main.py` 会按顺序执行 Task1-3
- 配置：所有模型和路径配置位于 `config/prediction_model_config.json`。

## 快速开始

1. 安装依赖

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. 在项目根目录运行完整流水线

```bash
python main.py
```

运行完成后，结果会生成到 `result/` 目录下，按任务分子目录保存指标、预测 CSV、图表与回测汇总

3. 运行单个脚本

```bash
# 调用某一个程序，这里以 K-means 为例
python src/task1/cluster_kmeans.py
```

**所有脚本都配置了 `""__main__""` 入口，可以单独运行**

## 配置要点

- 修改特征窗口、模型超参、滚动配置或输出路径 `config/prediction_model_config.json`
- `prediction_model.py` 通过 `set_split_ranges(...)` 传入自定义的训练/验证/测试区间

## 输出说明

- Task1：`data/stock_static.csv` 和 `result/task1/` 下的聚类/热力图等
- Task2：`result/task2/prediction_result_sp100/` 与 `result/task2/prediction_result_hsi/` 预测结果与对比图
- Task3：`result/task3/<window>/` 下每个滚动窗口的回测 `meta.json`、NAV CSV 与图，汇总 `result/task3/rolling_summary.csv`

## 注意事项

- 确保 `data/` 下原始 CSV 文件存在
- 确保运行目录在文件主入口
- 若使用交互式查看图像，将 `show_plot=True`；默认多数绘图仅保存到文件并不弹窗
