## Day27：通用机器学习流水线


### 学习目标
- 把“管道”当成一个固定流程来理解，弄清楚每一步的输入输出。
- 学习 transformer、estimator、pipeline 这三个关键词。
- 通过信贷违约数据，实现一次完整流水线。
- 总结出一个可以在其他机器学习任务里复用的通用 pipeline 模板。



### Pipeline 的三个角色
1. **转换器 Transformer**：它负责“整理数据”，只学习处理规则，不记住具体样本。典型动作是 `fit` + `transform`，比如缺失值填充、标准化、独热编码。Transformer 不会学习“预测目标”，但很多 Transformer 会从训练数据中学习统计量（均值、方差、中位数等）。
2. **估计器 Estimator**：它负责“学会预测”，既要 `fit`（找到模型参数）也要 `predict`。分类器、回归器都属于这一类。
3. **流水线 Pipeline**：把多个转换器和一个估计器按顺序串起来的“大盒子”。好处是：写一次流程，任意数据集都能重复同样的处理顺序，交叉验证或网格搜索时也能保证每个折叠都遵守同样的预处理，彻底杜绝信息泄露。



### 通用机器学习流水线的顺序
1. **明确目标**：任务类型、评估指标、业务约束。
2. **拿到原始数据**：只做必要的检查（缺失值、数据类型），不要贸然修改。
3. **划分训练集/测试集**：必须在任何预处理之前划分，防止数据泄露。
4. **定义预处理策略**：针对不同列（有序类别、无序类别、数值型等）准备对应的 transformer。
5. **装配 Pipeline 并训练**：把预处理和模型放进一个步骤列表操作，让 `fit` 自动完成全部动作。
6. **评估与调参**：同一个 pipeline 可以直接喂给交叉验证、网格搜索等调优工具。
7. **复用与部署**：把流水线当成一个整体对象持久化到磁盘，上线后只需 `predict`。



### 本节的数据与任务
我们继续沿用信贷违约数据。标签列 `Credit Default` 表示客户是否会违约，其他列既有连续变量（年收入、信用分等），也有有序/无序的分类变量（工作年限、贷款用途等）。这个数据结构很典型，适合演示如何一步步搭建通用流水线。



```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('data.csv')
print(f"原始数据形状: {data.shape}")
data.head()

```

    原始数据形状: (7500, 18)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Home Ownership</th>
      <th>Annual Income</th>
      <th>Years in current job</th>
      <th>Tax Liens</th>
      <th>Number of Open Accounts</th>
      <th>Years of Credit History</th>
      <th>Maximum Open Credit</th>
      <th>Number of Credit Problems</th>
      <th>Months since last delinquent</th>
      <th>Bankruptcies</th>
      <th>Purpose</th>
      <th>Term</th>
      <th>Current Loan Amount</th>
      <th>Current Credit Balance</th>
      <th>Monthly Debt</th>
      <th>Credit Score</th>
      <th>Credit Default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Own Home</td>
      <td>482087.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>26.3</td>
      <td>685960.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>debt consolidation</td>
      <td>Short Term</td>
      <td>99999999.0</td>
      <td>47386.0</td>
      <td>7914.0</td>
      <td>749.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Own Home</td>
      <td>1025487.0</td>
      <td>10+ years</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>15.3</td>
      <td>1181730.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>debt consolidation</td>
      <td>Long Term</td>
      <td>264968.0</td>
      <td>394972.0</td>
      <td>18373.0</td>
      <td>737.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Home Mortgage</td>
      <td>751412.0</td>
      <td>8 years</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>35.0</td>
      <td>1182434.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>debt consolidation</td>
      <td>Short Term</td>
      <td>99999999.0</td>
      <td>308389.0</td>
      <td>13651.0</td>
      <td>742.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Own Home</td>
      <td>805068.0</td>
      <td>6 years</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>22.5</td>
      <td>147400.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>debt consolidation</td>
      <td>Short Term</td>
      <td>121396.0</td>
      <td>95855.0</td>
      <td>11338.0</td>
      <td>694.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Rent</td>
      <td>776264.0</td>
      <td>8 years</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>13.6</td>
      <td>385836.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>debt consolidation</td>
      <td>Short Term</td>
      <td>125840.0</td>
      <td>93309.0</td>
      <td>7180.0</td>
      <td>719.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




### Step 1：先拆分特征和标签，并在任何处理之前划分数据
先把特征矩阵 `X` 和标签 `y` 分开，再立刻切分训练集与测试集。这样做是为了保证测试集始终像“真实新数据”，不会在预处理阶段被我们提前看过。



```python

X = data.drop('Credit Default', axis=1)
y = data['Credit Default']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n数据集划分完成。")
print("X_train 形状:", X_train.shape)
print("X_test 形状:", X_test.shape)
print("y_train 形状:", y_train.shape)
print("y_test 形状:", y_test.shape)

```


    数据集划分完成。
    X_train 形状: (6000, 17)
    X_test 形状: (1500, 17)
    y_train 形状: (6000,)
    y_test 形状: (1500,)



### Step 2：识别列类型并给出针对性的预处理策略
- `Home Ownership`、`Years in current job`、`Term` 是有序的分类变量，需要告诉模型先后顺序，所以使用 **OrdinalEncoder**。
- `Purpose` 是无序的分类变量，用 **OneHotEncoder** 做独热编码最安全。
- 其他列全部视为连续变量，用 **SimpleImputer** 补全缺失，再用 **StandardScaler** 做标准化，让不同量纲的特征在模型里地位相当。
- 这些转换器都只学习“处理规则”，不会记住训练数据，因此可以放心复用在测试集乃至线上数据上。



```python

ordinal_features = ['Home Ownership', 'Years in current job', 'Term']
ordinal_categories = [
    ['Own Home', 'Rent', 'Have Mortgage', 'Home Mortgage'],
    ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'],
    ['Short Term', 'Long Term']
]
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
])
print("有序特征处理 Pipeline 定义完成。")

# 标称分类特征 (对应你之前的独热编码)
nominal_features = ['Purpose']
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
print("标称特征处理 Pipeline 定义完成。")

# 构建处理连续特征的 Pipeline: 先填充缺失值，再进行标准化
numeric_features = [col for col in X.columns if col not in ordinal_features + nominal_features]
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
print("连续特征处理 Pipeline 定义完成。")

print('有序特征:', ordinal_features)
print('无序特征:', nominal_features)
print('连续特征数:', len(numeric_features))

```

    有序特征处理 Pipeline 定义完成。
    标称特征处理 Pipeline 定义完成。
    连续特征处理 Pipeline 定义完成。
    有序特征: ['Home Ownership', 'Years in current job', 'Term']
    无序特征: ['Purpose']
    连续特征数: 13



### Step 3：把预处理器和模型装进同一个 Pipeline
`ColumnTransformer` 负责把不同的 transformer 分别用在对应列上，再拼成新的特征矩阵；随后和模型一起交给 `Pipeline`。从此以后我们只需要调用一次 `fit`、`predict` 就能驱动整条流水线。



```python

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('nominal', nominal_transformer, nominal_features),
        ('numeric', numeric_transformer, numeric_features)
    ],
    remainder='drop'
    # 如何处理没有在上面列表中指定的列。
                           # 'passthrough' 表示保留这些列，不做任何处理。
                           # 'drop' 表示丢弃这些列。
)

rf_model = RandomForestClassifier(random_state=42)
credit_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', rf_model)
])
print("\n完整的 Pipeline 定义完成。")

```


    完整的 Pipeline 定义完成。



### Step 4：训练与评估
训练时，流水线会先在 `X_train` 上 `fit_transform` 预处理步骤，再 `fit` 模型；预测时同理。这里记录耗时、输出分类报告和混淆矩阵，方便与未来的改动做对比。



```python

start_time = time.time()

credit_pipeline.fit(X_train, y_train)
pred_test = credit_pipeline.predict(X_test)

end_time = time.time()

print(f"训练+预测耗时: {end_time-start_time:.4f} 秒")
print("测试集分类报告:", classification_report(y_test, pred_test))
print("测试集混淆矩阵:", confusion_matrix(y_test, pred_test))

```


### Step 5：把流程封装成可复用的模板
下面写一个辅助函数。它接收：
- 有序特征配置（特征名 + 类别顺序），
- 无序特征配置，
- 数值特征列表和希望使用的填充值策略，
- 以及任意一个 scikit-learn 模型。
这样不管换成回归任务还是别的分类器，都可以复用同一套预处理逻辑。



```python

def build_general_ml_pipeline(
    ordinal_cfg=None,
    nominal_cfg=None,
    numeric_cols=None,
    model=None,
    numeric_strategy='median'
):
    transformers = []

    if ordinal_cfg and ordinal_cfg.get('features'):
        categories = ordinal_cfg.get('categories')
        if categories is not None:
            ordinal_encoder = OrdinalEncoder(
                categories=categories,
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        else:
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
        ordinal_block = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', ordinal_encoder)
        ])
        transformers.append(('ordinal', ordinal_block, ordinal_cfg['features']))

    if nominal_cfg and nominal_cfg.get('features'):
        nominal_block = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('nominal', nominal_block, nominal_cfg['features']))

    if numeric_cols:
        numeric_block = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('numeric', numeric_block, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    final_model = model if model is not None else RandomForestClassifier()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', final_model)
    ])
    return pipeline


template_pipeline = build_general_ml_pipeline(
    ordinal_cfg={'features': ordinal_features, 'categories': ordinal_categories},
    nominal_cfg={'features': nominal_features},
    numeric_cols=numeric_features,
    model=RandomForestClassifier(n_estimators=200, random_state=42),
    numeric_strategy='median'
)

template_pipeline.fit(X_train, y_train)
template_pred = template_pipeline.predict(X_test)
print(f"模板 Pipeline 测试集准确率: {template_pipeline.score(X_test, y_test):.4f}")
print("模板版本分类报告:", classification_report(y_test, template_pred))

```

    模板 Pipeline 测试集准确率: 0.7640
    模板版本分类报告:               precision    recall  f1-score   support
    
               0       0.76      0.96      0.85      1059
               1       0.77      0.28      0.41       441
    
        accuracy                           0.76      1500
       macro avg       0.77      0.62      0.63      1500
    weighted avg       0.76      0.76      0.72      1500




### 小结
- 把 transformer、estimator 包进 Pipeline 后，只需要管理一个对象，能自动避免数据泄露，还能直接送交交叉验证/网格搜索。
- 为不同列自定义预处理策略是流水线的核心，提前列出有序/无序/数值特征能极大减少出错概率。
- 通用模板函数让我们在更换数据集或模型时只需修改少量配置。

