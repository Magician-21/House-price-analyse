
# 房价预测模型开发文档

## 一、项目概述
本项目基于Kaggle经典房价预测数据集（House Prices: Advanced Regression Techniques），通过数据预处理、特征工程和机器学习建模，实现对住宅销售价格的预测。项目涵盖数据清洗、缺失值处理、特征编码、特征工程、模型训练与调优全流程，最终使用随机森林回归模型生成预测结果。


## 二、数据集说明
### 数据来源
- 训练集：`train.csv`，包含1460条样本，81个特征（含目标变量`SalePrice`）
- 测试集：`test.csv`，包含1459条样本，80个特征（无`SalePrice`）
- 目标变量：`SalePrice`（连续型数值，需预测）

### 关键特征说明
- 数值型特征：如`LotFrontage`（临街宽度）、`GrLivArea`（居住面积）、`TotalBsmtSF`（地下室总面积）等
- 分类型特征：如`MSZoning`（用地类型）、`Neighborhood`（社区）、`KitchenQual`（厨房质量）等
- 时间相关特征：`YearBuilt`（建造年份）、`YrSold`（出售年份）等


## 三、环境依赖
### 主要库版本
```python
python >= 3.8
pandas >= 1.3.3
numpy >= 1.21.2
scikit-learn >= 1.0.2
seaborn >= 0.11.2
matplotlib >= 3.4.3
```

### 安装命令
```bash
pip install -r requirements.txt  # 若存在依赖文件
# 或手动安装
pip install pandas numpy scikit-learn seaborn matplotlib
```


## 四、代码结构与功能说明
### 1. 数据预处理
#### 缺失值处理
- **高缺失率特征删除**：删除缺失率>30%的特征（如`PoolQC`、`Alley`、`Fence`等）
- **数值型缺失值填充**：
  - `LotFrontage`：按`MSSubClass`分组填充中位数
  - `GarageYrBlt`：填充0
  - `MasVnrArea`：填充0
- **分类型缺失值填充**：
  - `GarageType`、`BsmtQual`等：填充`'None'`
  - 测试集分类特征用训练集众数填充

#### 异常值处理
- 未显式处理，通过模型鲁棒性（如随机森林）隐式处理

### 2. 特征工程
#### 特征编码
- **标签编码（Label Encoding）**：类别数较少的无序特征（如`Street`、`CentralAir`）
- **独热编码（One-Hot Encoding）**：高基数类别特征（如`Neighborhood`、`HouseStyle`）
- **目标编码（Mean Encoding）**：`GarageType`使用目标变量均值编码

#### 特征生成
- **时间特征**：`HouseAge = YrSold - YearBuilt`，`RemodelAge = YrSold - YearRemodAdd`
- **组合特征**：
  - `TotalBsmtSF = BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF`
  - `TotalRooms = TotRmsAbvGrd + KitchenAbvGr`（简化版，实际代码中可能有调整）

#### 特征选择
- **卡方检验**：删除与目标变量无关的高频率分类特征
- **递归特征消除（RFE）**：选择20个关键特征用于模型训练

### 3. 模型训练与调优
#### 模型选择
- **基线模型**：线性回归（Linear Regression）
- **最终模型**：随机森林回归（Random Forest Regressor），因其在交叉验证中表现最优（RMSE更低）

#### 超参数调优
- 使用`RandomizedSearchCV`进行随机搜索，优化参数包括：
  - `n_estimators`：50/100/200/300
  - `max_depth`：None/10/20/30
  - `min_samples_split`：2/5/10
  - `min_samples_leaf`：1/2/4
- **最佳超参数**：`n_estimators=100, max_depth=30, min_samples_split=10, min_samples_leaf=1`

#### 评估指标
- 均方根误差（RMSE）：交叉验证平均RMSE为34,474.04（随机森林模型）


## 五、使用指南
### 1. 数据准备
- 将训练集和测试集命名为`train.csv`和`test.csv`，放入项目根目录
- 目录结构：
  ```
  project/
  ├── train.csv
  ├── test.csv
  ├── code.py        # 主代码文件
  └── submission.csv # 预测结果文件
  ```

### 2. 运行步骤
#### 步骤1：数据预处理与特征工程
```python
# 执行数据清洗、特征编码、特征生成等步骤
# 代码见`data_preprocessing.py`或主代码文件前半部分
```

#### 步骤2：模型训练与预测
```python
# 训练随机森林模型并生成预测结果
python code.py
```

#### 步骤3：生成提交文件
- 预测结果保存至`submission.csv`，格式：
  ```csv
  Id,SalePrice
  1461,180000
  1462,200000
  ...
  ```


## 六、结果展示
### 关键指标
| 模型类型          | 交叉验证RMSE | 测试集RMSE   |
|-------------------|--------------|--------------|
| 线性回归         | 43,230.05    | 未记录       |
| 随机森林（调优后）| 34,474.04    | 约35,000     |

### 特征重要性
- 前5大重要特征（随机森林模型）：
  1. `OverallQual`（整体质量）
  2. `GrLivArea`（居住面积）
  3. `GarageCars`（车库容量）
  4. `TotalBsmtSF`（地下室总面积）
  5. `YearBuilt`（建造年份）


## 七、文件说明
- `code.py`：主代码文件，包含完整数据处理和建模流程
- `submission_model.pkl`：训练好的随机森林模型（通过`joblib`保存）
- `submission.csv`：最终预测结果文件


## 八、贡献与反馈
- 如需改进模型，可尝试：
  - 调整特征工程策略（如添加交互特征）
  - 尝试其他模型（如XGBoost、LightGBM）
  - 进一步调优超参数
- 反馈问题或建议：[提交Issue](https://github.com/Magician-21/House-price-analyse/issues)


## 九、许可证
本项目采用MIT许可证，允许自由修改和分发。见`LICENSE`文件。


**项目作者**：Klein
**更新日期**：2025年5月25日
