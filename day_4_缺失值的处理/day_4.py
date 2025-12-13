import pandas as pd

# 读取数据
data_csv =pd.read_csv('data.csv')

# 打印信息
print(data_csv)
print(data_csv.shape)
print(data_csv.columns)
print(data_csv.isnull().sum())

# 中位数填补空缺
median_income=data_csv['Annual Income'].median()
data_csv['Annual Income'].fillna(median_income,inplace=True)
print(data_csv['Annual Income'].isnull().sum())

# 众数填补空缺
mode=data_csv['Annual Income'].mode()
print(mode)
mode=mode[0]
data_csv['Annual Income'].fillna(mode,inplace=True)
print(data_csv['Annual Income'].isnull().sum())

# 循环补全所有列空值
c=data_csv.columns.tolist()
for i in c:
    if data_csv[i].dtype!='object':
        if data_csv[i].isnull().sum()>0:
            data_csv[i].fillna(data_csv[i].mean(),inplace=True)
print(data_csv.isnull().sum())