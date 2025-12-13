# Day 26 回归模型评价：熵权法 + TOPSIS

之前对机器学习模型进行评估时，常常面临多个指标拉扯的问题，很难只凭经验判断谁更优秀。

本节我们继续沿用多目标优化的思路，引入更系统的评价问题方法论，用熵权法结合 TOPSIS 来帮助我们为模型打分。

数据集为：加州房价数据集

- **指标的冲突性**：MSE、MAE 越低越好，但 R² 越高越好，还有训练时间这种成本维度，不可能同时最好。
- **主观性风险**：如果仅凭直觉设定权重，结论会随人而异。
- **缺乏统一标准**：传统方式只盯某一个指标（如 R²），无法兼顾整体表现。

**熵权法（Entropy Weight Method）**：依据指标数据的离散程度自动赋权，差异越大说明区分度越好，权重越高。
**TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**：在权重基础上，衡量每个模型与“理想解”和“负理想解”的距离，得到最终排序。整个流程可拆为“客观赋权 + 综合评价”两大阶段。

## 1. 数据预处理



```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style='whitegrid', font='SimHei')

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
df.columns = [
    'MedInc (中位收入)', 'HouseAge (房龄)', 'AveRooms (平均房间数)',
    'AveBedrms (平均卧室数)', 'Population (人口)', 'AveOccup (平均居住人数)',
    'Latitude (纬度)', 'Longitude (经度)', 'MedHouseVal (房价中位数)']
print(f"样本量：{df.shape[0]}，特征数：{df.shape[1]-1}")
df.head()
```

    样本量：20640，特征数：8





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
      <th>MedInc (中位收入)</th>
      <th>HouseAge (房龄)</th>
      <th>AveRooms (平均房间数)</th>
      <th>AveBedrms (平均卧室数)</th>
      <th>Population (人口)</th>
      <th>AveOccup (平均居住人数)</th>
      <th>Latitude (纬度)</th>
      <th>Longitude (经度)</th>
      <th>MedHouseVal (房价中位数)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集：{X_train.shape}, 测试集：{X_test.shape}")
```

    训练集：(16512, 8), 测试集：(4128, 8)


## 2. 构建待比较的回归模型

挑选四个常见的基准模型（线性回归、决策树、随机森林、梯度提升）来模拟真实建模场景。


```python
regressors = {
    'Linear Regression (线性回归)': LinearRegression(),
    'Decision Tree (决策树)': DecisionTreeRegressor(random_state=42),
    'Random Forest (随机森林)': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting (梯度提升)': GradientBoostingRegressor(n_estimators=100, random_state=42)
}
```

## 3. 训练模型并收集多维指标

我们记录每个模型在测试集上的 **MSE / RMSE / MAE / R²** 以及训练耗时，构建原始决策矩阵 `results_df`。


```python
records = []
for name, model in regressors.items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    duration = time.perf_counter() - start

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    records.append({
        'Model': name,
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'Mean Absolute Error (MAE)': mae,
        'R2 Score': r2,
        'Training Time (s)': duration
    })

results_df = pd.DataFrame(records)
results_df
```




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
      <th>Model</th>
      <th>Mean Squared Error (MSE)</th>
      <th>Root Mean Squared Error (RMSE)</th>
      <th>Mean Absolute Error (MAE)</th>
      <th>R2 Score</th>
      <th>Training Time (s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Linear Regression (线性回归)</td>
      <td>0.555892</td>
      <td>0.745581</td>
      <td>0.533200</td>
      <td>0.575788</td>
      <td>0.019426</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree (决策树)</td>
      <td>0.495235</td>
      <td>0.703729</td>
      <td>0.454679</td>
      <td>0.622076</td>
      <td>0.139013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest (随机森林)</td>
      <td>0.255368</td>
      <td>0.505340</td>
      <td>0.327543</td>
      <td>0.805123</td>
      <td>8.383994</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gradient Boosting (梯度提升)</td>
      <td>0.293997</td>
      <td>0.542215</td>
      <td>0.371643</td>
      <td>0.775645</td>
      <td>2.616082</td>
    </tr>
  </tbody>
</table>
</div>



我们已经拿到了 `results_df`（原始决策矩阵），接下来仍旧是三部曲：**数据标准化 → 熵权计算 → TOPSIS 排序**。

## 4. 指标方向与数据预处理

机器不知道“误差越小越好、R² 越大越好”，必须手动标明指标属性。训练时间同样视为成本型指标。


```python
benefit_cols = ['R2 Score']
cost_cols = ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)',
             'Mean Absolute Error (MAE)', 'Training Time (s)']

data_eval = results_df.set_index('Model')[benefit_cols + cost_cols].astype(float)

print('步骤 1 完成：指标方向已明确定义。')
print(f'效益型指标 (+): {benefit_cols}')
print(f'成本型指标 (-): {cost_cols}')
```

    步骤 1 完成：指标方向已明确定义。
    效益型指标 (+): ['R2 Score']
    成本型指标 (-): ['Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'Training Time (s)']


## 5. 数据标准化 (Normalization)

为避免不同量纲的影响，效益型指标采用 $(x-\min)/(\max-\min)$，成本型指标采用 $(\max-x)/(\max-\min)$，最后加上极小量 $\epsilon$ 防止 $\ln(0)$。


```python
epsilon = 1e-6

for col in benefit_cols:
    min_val = data_eval[col].min()
    max_val = data_eval[col].max()
    data_eval[col] = 1.0 if max_val == min_val else (data_eval[col] - min_val) / (max_val - min_val)

for col in cost_cols:
    min_val = data_eval[col].min()
    max_val = data_eval[col].max()
    data_eval[col] = 1.0 if max_val == min_val else (max_val - data_eval[col]) / (max_val - min_val)

data_eval = data_eval + epsilon
print('步骤 2 完成：指标已标准化。')
data_eval
```

    步骤 2 完成：指标已标准化。





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
      <th>R2 Score</th>
      <th>Mean Squared Error (MSE)</th>
      <th>Root Mean Squared Error (RMSE)</th>
      <th>Mean Absolute Error (MAE)</th>
      <th>Training Time (s)</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Linear Regression (线性回归)</th>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>0.000001</td>
      <td>1.000001</td>
    </tr>
    <tr>
      <th>Decision Tree (决策树)</th>
      <td>0.201837</td>
      <td>0.201837</td>
      <td>0.174209</td>
      <td>0.381805</td>
      <td>0.985704</td>
    </tr>
    <tr>
      <th>Random Forest (随机森林)</th>
      <td>1.000001</td>
      <td>1.000001</td>
      <td>1.000001</td>
      <td>1.000001</td>
      <td>0.000001</td>
    </tr>
    <tr>
      <th>Gradient Boosting (梯度提升)</th>
      <td>0.871462</td>
      <td>0.871462</td>
      <td>0.846509</td>
      <td>0.785567</td>
      <td>0.689566</td>
    </tr>
  </tbody>
</table>
</div>



## 6. 熵权法计算权重

指标差异越大说明越能区分模型，其权重应更高。信息熵提供了量化的依据。


```python
n, m = data_eval.shape
P = data_eval.div(data_eval.sum(axis=0), axis=1)
k = 1 / np.log(n)
E = -k * (P * np.log(P)).sum(axis=0)
d = 1 - E
weights = d / d.sum()

weights_df = pd.DataFrame(weights, columns=['Weight']).sort_values('Weight', ascending=False)
weights_df
```




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
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Root Mean Squared Error (RMSE)</th>
      <td>0.230469</td>
    </tr>
    <tr>
      <th>R2 Score</th>
      <td>0.221074</td>
    </tr>
    <tr>
      <th>Mean Squared Error (MSE)</th>
      <td>0.221074</td>
    </tr>
    <tr>
      <th>Mean Absolute Error (MAE)</th>
      <td>0.177287</td>
    </tr>
    <tr>
      <th>Training Time (s)</th>
      <td>0.150096</td>
    </tr>
  </tbody>
</table>
</div>



## 7. TOPSIS 综合评价

- **加权**：把每列指标乘以对应权重。
- **找标杆**：每列最大值是理想解，最小值是负理想解。
- **算距离**：求模型到理想解/负理想解的欧氏距离。
- **打分**：计算相对接近度 $C_i$，越接近 1 表示越优秀。


```python
V = data_eval * weights
V_plus = V.max()
V_minus = V.min()

D_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))
D_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))

scores = D_minus / (D_plus + D_minus)

final_results = results_df.copy()
final_results['TOPSIS Score'] = final_results['Model'].map(scores)
final_results['Rank'] = final_results['TOPSIS Score'].rank(ascending=False).astype(int)
final_results = final_results.sort_values('Rank')

columns_to_show = ['Model', 'R2 Score', 'Mean Squared Error (MSE)',
                    'Mean Absolute Error (MAE)', 'Training Time (s)',
                    'TOPSIS Score', 'Rank']
final_results[columns_to_show]
```




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
      <th>Model</th>
      <th>R2 Score</th>
      <th>Mean Squared Error (MSE)</th>
      <th>Mean Absolute Error (MAE)</th>
      <th>Training Time (s)</th>
      <th>TOPSIS Score</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Gradient Boosting (梯度提升)</td>
      <td>0.775645</td>
      <td>0.293997</td>
      <td>0.371643</td>
      <td>2.616082</td>
      <td>0.824156</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Random Forest (随机森林)</td>
      <td>0.805123</td>
      <td>0.255368</td>
      <td>0.327543</td>
      <td>8.383994</td>
      <td>0.739894</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Decision Tree (决策树)</td>
      <td>0.622076</td>
      <td>0.495235</td>
      <td>0.454679</td>
      <td>0.139013</td>
      <td>0.350084</td>
      <td>3</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Linear Regression (线性回归)</td>
      <td>0.575788</td>
      <td>0.555892</td>
      <td>0.533200</td>
      <td>0.019426</td>
      <td>0.260106</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



从结果可以看出，综合考虑误差、拟合优度与训练成本后，树模型（随机森林、梯度提升）往往能取得更高的 TOPSIS 得分。该流程可平滑迁移到任何回归任务，只需替换数据和指标即可复用。
