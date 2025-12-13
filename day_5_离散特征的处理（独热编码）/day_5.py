import pandas as pd

data=pd.read_csv('data.csv')

# 填补缺失值
for column in data.columns:
    if data[column].isnull().sum() > 0:
        if data[column].dtype !='object':
            # 均值填补缺失
            data[column].fillna(data[column].mean(),inplace=True)
        else:
            # 众数填补缺失的object对象
            data[column].fillna(data[column].mode()[0],inplace=True)

# 将object对象独热编码
discrete_list=[]
for column in data.columns:
    if data[column].dtype=='object':
        print(f'{column}类数据分布情况：')
        print(data[column].value_counts())
        print('----------------------------------')
        discrete_list.append(column)
data=pd.get_dummies(data,columns=discrete_list,drop_first=True)

# 将布尔值转换为整型
for column in data.columns:
    if data[column].dtype=='bool':
        data[column]=data[column].astype(int)

print('最终结果为：')
print(data)
