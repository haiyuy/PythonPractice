# DAY 37 · GPU训练与 __call__ 方法

- 重新跑通鸢尾花分类任务并记录 CPU 运行时长
- 学习如何快速了解硬件配置，尤其是 CPU 与 GPU
- 掌握在 PyTorch 中将数据和模型迁移到 GPU 的常见做法
- 分析 GPU 看起来更慢的根源并给出优化策略
- 理解 nn.Module 中 __call__ 的工作机制


## 1. 在 CPU 上搭建基线

先用 CPU 完成一次完整训练流程，包含数据预处理、模型定义、训练循环和耗时统计。


```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 仍然用4特征，3分类的鸢尾花数据集作为我们今天的数据集
# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 归一化数据，神经网络对于输入数据的尺寸敏感，归一化是最常见的处理方式
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) #确保训练集和测试集是相同的缩放

# 将数据转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 20000
losses = []

import time
start_time = time.time()

for epoch in range(num_epochs):
    outputs = model.forward(X_train)   # 显式调用forward函数
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

time_all = time.time() - start_time
print(f'Training time: {time_all:.2f} seconds')

import matplotlib.pyplot as plt
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

```

    Epoch [100/20000], Loss: 1.0882
    Epoch [200/20000], Loss: 1.0701
    Epoch [300/20000], Loss: 1.0455
    Epoch [400/20000], Loss: 1.0111
    Epoch [500/20000], Loss: 0.9649
    Epoch [600/20000], Loss: 0.9090
    Epoch [700/20000], Loss: 0.8466
    Epoch [800/20000], Loss: 0.7813
    Epoch [900/20000], Loss: 0.7184
    Epoch [1000/20000], Loss: 0.6617
    Epoch [1100/20000], Loss: 0.6130
    Epoch [1200/20000], Loss: 0.5721
    Epoch [1300/20000], Loss: 0.5380
    Epoch [1400/20000], Loss: 0.5092
    Epoch [1500/20000], Loss: 0.4845
    Epoch [1600/20000], Loss: 0.4630
    Epoch [1700/20000], Loss: 0.4439
    Epoch [1800/20000], Loss: 0.4265
    Epoch [1900/20000], Loss: 0.4107
    Epoch [2000/20000], Loss: 0.3959
    Epoch [2100/20000], Loss: 0.3821
    Epoch [2200/20000], Loss: 0.3691
    Epoch [2300/20000], Loss: 0.3568
    Epoch [2400/20000], Loss: 0.3451
    Epoch [2500/20000], Loss: 0.3339
    Epoch [2600/20000], Loss: 0.3233
    Epoch [2700/20000], Loss: 0.3131
    Epoch [2800/20000], Loss: 0.3033
    Epoch [2900/20000], Loss: 0.2940
    Epoch [3000/20000], Loss: 0.2851
    Epoch [3100/20000], Loss: 0.2766
    Epoch [3200/20000], Loss: 0.2684
    Epoch [3300/20000], Loss: 0.2605
    Epoch [3400/20000], Loss: 0.2531
    Epoch [3500/20000], Loss: 0.2459
    Epoch [3600/20000], Loss: 0.2390
    Epoch [3700/20000], Loss: 0.2325
    Epoch [3800/20000], Loss: 0.2262
    Epoch [3900/20000], Loss: 0.2202
    Epoch [4000/20000], Loss: 0.2145
    Epoch [4100/20000], Loss: 0.2091
    Epoch [4200/20000], Loss: 0.2038
    Epoch [4300/20000], Loss: 0.1988
    Epoch [4400/20000], Loss: 0.1941
    Epoch [4500/20000], Loss: 0.1895
    Epoch [4600/20000], Loss: 0.1852
    Epoch [4700/20000], Loss: 0.1810
    Epoch [4800/20000], Loss: 0.1770
    Epoch [4900/20000], Loss: 0.1732
    Epoch [5000/20000], Loss: 0.1695
    Epoch [5100/20000], Loss: 0.1660
    Epoch [5200/20000], Loss: 0.1627
    Epoch [5300/20000], Loss: 0.1595
    Epoch [5400/20000], Loss: 0.1564
    Epoch [5500/20000], Loss: 0.1534
    Epoch [5600/20000], Loss: 0.1506
    Epoch [5700/20000], Loss: 0.1479
    Epoch [5800/20000], Loss: 0.1452
    Epoch [5900/20000], Loss: 0.1427
    Epoch [6000/20000], Loss: 0.1403
    Epoch [6100/20000], Loss: 0.1380
    Epoch [6200/20000], Loss: 0.1358
    Epoch [6300/20000], Loss: 0.1336
    Epoch [6400/20000], Loss: 0.1316
    Epoch [6500/20000], Loss: 0.1296
    Epoch [6600/20000], Loss: 0.1277
    Epoch [6700/20000], Loss: 0.1258
    Epoch [6800/20000], Loss: 0.1240
    Epoch [6900/20000], Loss: 0.1223
    Epoch [7000/20000], Loss: 0.1207
    Epoch [7100/20000], Loss: 0.1191
    Epoch [7200/20000], Loss: 0.1175
    Epoch [7300/20000], Loss: 0.1160
    Epoch [7400/20000], Loss: 0.1146
    Epoch [7500/20000], Loss: 0.1132
    Epoch [7600/20000], Loss: 0.1119
    Epoch [7700/20000], Loss: 0.1106
    Epoch [7800/20000], Loss: 0.1093
    Epoch [7900/20000], Loss: 0.1081
    Epoch [8000/20000], Loss: 0.1069
    Epoch [8100/20000], Loss: 0.1058
    Epoch [8200/20000], Loss: 0.1047
    Epoch [8300/20000], Loss: 0.1036
    Epoch [8400/20000], Loss: 0.1026
    Epoch [8500/20000], Loss: 0.1015
    Epoch [8600/20000], Loss: 0.1006
    Epoch [8700/20000], Loss: 0.0996
    Epoch [8800/20000], Loss: 0.0987
    Epoch [8900/20000], Loss: 0.0978
    Epoch [9000/20000], Loss: 0.0969
    Epoch [9100/20000], Loss: 0.0961
    Epoch [9200/20000], Loss: 0.0953
    Epoch [9300/20000], Loss: 0.0945
    Epoch [9400/20000], Loss: 0.0937
    Epoch [9500/20000], Loss: 0.0930
    Epoch [9600/20000], Loss: 0.0922
    Epoch [9700/20000], Loss: 0.0915
    Epoch [9800/20000], Loss: 0.0908
    Epoch [9900/20000], Loss: 0.0902
    Epoch [10000/20000], Loss: 0.0895
    Epoch [10100/20000], Loss: 0.0889
    Epoch [10200/20000], Loss: 0.0882
    Epoch [10300/20000], Loss: 0.0876
    Epoch [10400/20000], Loss: 0.0870
    Epoch [10500/20000], Loss: 0.0865
    Epoch [10600/20000], Loss: 0.0859
    Epoch [10700/20000], Loss: 0.0854
    Epoch [10800/20000], Loss: 0.0848
    Epoch [10900/20000], Loss: 0.0843
    Epoch [11000/20000], Loss: 0.0838
    Epoch [11100/20000], Loss: 0.0833
    Epoch [11200/20000], Loss: 0.0828
    Epoch [11300/20000], Loss: 0.0823
    Epoch [11400/20000], Loss: 0.0819
    Epoch [11500/20000], Loss: 0.0814
    Epoch [11600/20000], Loss: 0.0810
    Epoch [11700/20000], Loss: 0.0805
    Epoch [11800/20000], Loss: 0.0801
    Epoch [11900/20000], Loss: 0.0797
    Epoch [12000/20000], Loss: 0.0793
    Epoch [12100/20000], Loss: 0.0789
    Epoch [12200/20000], Loss: 0.0785
    Epoch [12300/20000], Loss: 0.0781
    Epoch [12400/20000], Loss: 0.0778
    Epoch [12500/20000], Loss: 0.0774
    Epoch [12600/20000], Loss: 0.0770
    Epoch [12700/20000], Loss: 0.0767
    Epoch [12800/20000], Loss: 0.0764
    Epoch [12900/20000], Loss: 0.0760
    Epoch [13000/20000], Loss: 0.0757
    Epoch [13100/20000], Loss: 0.0754
    Epoch [13200/20000], Loss: 0.0751
    Epoch [13300/20000], Loss: 0.0747
    Epoch [13400/20000], Loss: 0.0744
    Epoch [13500/20000], Loss: 0.0741
    Epoch [13600/20000], Loss: 0.0739
    Epoch [13700/20000], Loss: 0.0736
    Epoch [13800/20000], Loss: 0.0733
    Epoch [13900/20000], Loss: 0.0730
    Epoch [14000/20000], Loss: 0.0727
    Epoch [14100/20000], Loss: 0.0725
    Epoch [14200/20000], Loss: 0.0722
    Epoch [14300/20000], Loss: 0.0720
    Epoch [14400/20000], Loss: 0.0717
    Epoch [14500/20000], Loss: 0.0715
    Epoch [14600/20000], Loss: 0.0712
    Epoch [14700/20000], Loss: 0.0710
    Epoch [14800/20000], Loss: 0.0707
    Epoch [14900/20000], Loss: 0.0705
    Epoch [15000/20000], Loss: 0.0703
    Epoch [15100/20000], Loss: 0.0701
    Epoch [15200/20000], Loss: 0.0698
    Epoch [15300/20000], Loss: 0.0696
    Epoch [15400/20000], Loss: 0.0694
    Epoch [15500/20000], Loss: 0.0692
    Epoch [15600/20000], Loss: 0.0690
    Epoch [15700/20000], Loss: 0.0688
    Epoch [15800/20000], Loss: 0.0686
    Epoch [15900/20000], Loss: 0.0684
    Epoch [16000/20000], Loss: 0.0682
    Epoch [16100/20000], Loss: 0.0680
    Epoch [16200/20000], Loss: 0.0678
    Epoch [16300/20000], Loss: 0.0677
    Epoch [16400/20000], Loss: 0.0675
    Epoch [16500/20000], Loss: 0.0673
    Epoch [16600/20000], Loss: 0.0671
    Epoch [16700/20000], Loss: 0.0670
    Epoch [16800/20000], Loss: 0.0668
    Epoch [16900/20000], Loss: 0.0666
    Epoch [17000/20000], Loss: 0.0664
    Epoch [17100/20000], Loss: 0.0663
    Epoch [17200/20000], Loss: 0.0661
    Epoch [17300/20000], Loss: 0.0660
    Epoch [17400/20000], Loss: 0.0658
    Epoch [17500/20000], Loss: 0.0657
    Epoch [17600/20000], Loss: 0.0655
    Epoch [17700/20000], Loss: 0.0654
    Epoch [17800/20000], Loss: 0.0652
    Epoch [17900/20000], Loss: 0.0651
    Epoch [18000/20000], Loss: 0.0649
    Epoch [18100/20000], Loss: 0.0648
    Epoch [18200/20000], Loss: 0.0646
    Epoch [18300/20000], Loss: 0.0645
    Epoch [18400/20000], Loss: 0.0644
    Epoch [18500/20000], Loss: 0.0642
    Epoch [18600/20000], Loss: 0.0641
    Epoch [18700/20000], Loss: 0.0640
    Epoch [18800/20000], Loss: 0.0638
    Epoch [18900/20000], Loss: 0.0637
    Epoch [19000/20000], Loss: 0.0636
    Epoch [19100/20000], Loss: 0.0635
    Epoch [19200/20000], Loss: 0.0633
    Epoch [19300/20000], Loss: 0.0632
    Epoch [19400/20000], Loss: 0.0631
    Epoch [19500/20000], Loss: 0.0630
    Epoch [19600/20000], Loss: 0.0629
    Epoch [19700/20000], Loss: 0.0628
    Epoch [19800/20000], Loss: 0.0626
    Epoch [19900/20000], Loss: 0.0625
    Epoch [20000/20000], Loss: 0.0624
    Training time: 4.62 seconds



    
![png](Day_37_files/Day_37_2_1.png)
    


### 1.1 查看 CPU 指标

上述是在cpu的情况下训练，（即使安装了cuda，但是没有使用cuda），我们借这个机会简单介绍下cpu的性能差异。


我使用的是wsl。因此我在终端输入lscpu来查看CPU信息。
```bash
(base) ubuntu24@DESKTOP-3Q94GS2:~/code/PythonPractice/day_37$ lscpu
CPU(s):                   24
Thread(s) per core:       2
Core(s) per socket:       12
Socket(s):                1
```
![image.png](Day_37_files/image.png)
![image-2.png](Day_37_files/image-2.png)



**CPU 配置解读（i7-12800HX）**

* Intel 第 12 代酷睿（Alder Lake）移动端高性能处理器
* 核心架构：混合大小核设计

  * 性能核（P-Core）：8 核，支持超线程（16 线程）
  * 能效核（E-Core）：4 核，不支持超线程
* 物理核心数：12 核
* 逻辑线程数：24




在该配置下，CPU 上的鸢尾花训练平均约 4.6 秒。接下来开始研究 GPU：如何迁移模型、如何评估显卡、又该如何正确理解速度差异。


## 2. GPU 训练

在 PyTorch 中，`.to(device)` 可以把张量或模型移动到指定设备（CPU/GPU）。只有 `torch.Tensor` 对象和继承 `nn.Module` 的模型拥有该方法。实践中需要保证输入张量和模型在同一设备上，否则会抛出运行时错误。


### 2.1 如何快速看懂 GPU 型号

以 RTX 3090 Ti、RTX 3080、RTX 3070 Ti、RTX 4070 等为例：

- **代际**：前两位数字代表代数，40xx 为第 40 代、30xx 为第 30 代。新架构通常意味着更先进的制程和更高的能效比。
- **级别**：后两位数字代表定位。
    - xx90：旗舰/次旗舰，性能最强、显存最大。
    - xx80：高端，性能强劲、显存较多。
    - xx70：中高端，适合兼顾训练和日常使用。



```python
import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    print("CUDA可用！")
    device_count = torch.cuda.device_count()
    print(f"可用的CUDA设备数量: {device_count}")
    current_device = torch.cuda.current_device()
    print(f"当前使用的CUDA设备索引: {current_device}")
    device_name = torch.cuda.get_device_name(current_device)
    print(f"当前CUDA设备的名称: {device_name}")
    cuda_version = torch.version.cuda
    print(f"CUDA版本: {cuda_version}")
    print("cuDNN版本:", torch.backends.cudnn.version())
else:
    print("CUDA不可用。")

```

    CUDA可用！
    可用的CUDA设备数量: 1
    当前使用的CUDA设备索引: 0
    当前CUDA设备的名称: NVIDIA GeForce RTX 4070 Laptop GPU
    CUDA版本: 12.4
    cuDNN版本: 90100



```python
# 设置GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

```

    使用设备: cuda:0



```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 归一化数据
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为PyTorch张量并移至GPU
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

```


```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型并移至GPU
model = MLP().to(device)

```


```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 20000
losses = []

start_time = time.time()

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

time_all = time.time() - start_time
print(f'Training time: {time_all:.2f} seconds')

plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

```

    Epoch [100/20000], Loss: 1.0390
    Epoch [200/20000], Loss: 0.9741
    Epoch [300/20000], Loss: 0.9003
    Epoch [400/20000], Loss: 0.8244
    Epoch [500/20000], Loss: 0.7507
    Epoch [600/20000], Loss: 0.6835
    Epoch [700/20000], Loss: 0.6255
    Epoch [800/20000], Loss: 0.5776
    Epoch [900/20000], Loss: 0.5386
    Epoch [1000/20000], Loss: 0.5066
    Epoch [1100/20000], Loss: 0.4798
    Epoch [1200/20000], Loss: 0.4570
    Epoch [1300/20000], Loss: 0.4372
    Epoch [1400/20000], Loss: 0.4197
    Epoch [1500/20000], Loss: 0.4039
    Epoch [1600/20000], Loss: 0.3894
    Epoch [1700/20000], Loss: 0.3760
    Epoch [1800/20000], Loss: 0.3635
    Epoch [1900/20000], Loss: 0.3517
    Epoch [2000/20000], Loss: 0.3405
    Epoch [2100/20000], Loss: 0.3298
    Epoch [2200/20000], Loss: 0.3196
    Epoch [2300/20000], Loss: 0.3099
    Epoch [2400/20000], Loss: 0.3006
    Epoch [2500/20000], Loss: 0.2917
    Epoch [2600/20000], Loss: 0.2831
    Epoch [2700/20000], Loss: 0.2749
    Epoch [2800/20000], Loss: 0.2671
    Epoch [2900/20000], Loss: 0.2596
    Epoch [3000/20000], Loss: 0.2524
    Epoch [3100/20000], Loss: 0.2456
    Epoch [3200/20000], Loss: 0.2391
    Epoch [3300/20000], Loss: 0.2328
    Epoch [3400/20000], Loss: 0.2268
    Epoch [3500/20000], Loss: 0.2211
    Epoch [3600/20000], Loss: 0.2157
    Epoch [3700/20000], Loss: 0.2104
    Epoch [3800/20000], Loss: 0.2054
    Epoch [3900/20000], Loss: 0.2006
    Epoch [4000/20000], Loss: 0.1960
    Epoch [4100/20000], Loss: 0.1916
    Epoch [4200/20000], Loss: 0.1874
    Epoch [4300/20000], Loss: 0.1834
    Epoch [4400/20000], Loss: 0.1795
    Epoch [4500/20000], Loss: 0.1758
    Epoch [4600/20000], Loss: 0.1722
    Epoch [4700/20000], Loss: 0.1688
    Epoch [4800/20000], Loss: 0.1655
    Epoch [4900/20000], Loss: 0.1624
    Epoch [5000/20000], Loss: 0.1593
    Epoch [5100/20000], Loss: 0.1564
    Epoch [5200/20000], Loss: 0.1536
    Epoch [5300/20000], Loss: 0.1509
    Epoch [5400/20000], Loss: 0.1483
    Epoch [5500/20000], Loss: 0.1458
    Epoch [5600/20000], Loss: 0.1434
    Epoch [5700/20000], Loss: 0.1411
    Epoch [5800/20000], Loss: 0.1389
    Epoch [5900/20000], Loss: 0.1367
    Epoch [6000/20000], Loss: 0.1346
    Epoch [6100/20000], Loss: 0.1327
    Epoch [6200/20000], Loss: 0.1307
    Epoch [6300/20000], Loss: 0.1289
    Epoch [6400/20000], Loss: 0.1271
    Epoch [6500/20000], Loss: 0.1253
    Epoch [6600/20000], Loss: 0.1237
    Epoch [6700/20000], Loss: 0.1220
    Epoch [6800/20000], Loss: 0.1205
    Epoch [6900/20000], Loss: 0.1190
    Epoch [7000/20000], Loss: 0.1175
    Epoch [7100/20000], Loss: 0.1161
    Epoch [7200/20000], Loss: 0.1147
    Epoch [7300/20000], Loss: 0.1134
    Epoch [7400/20000], Loss: 0.1121
    Epoch [7500/20000], Loss: 0.1109
    Epoch [7600/20000], Loss: 0.1097
    Epoch [7700/20000], Loss: 0.1085
    Epoch [7800/20000], Loss: 0.1074
    Epoch [7900/20000], Loss: 0.1063
    Epoch [8000/20000], Loss: 0.1052
    Epoch [8100/20000], Loss: 0.1042
    Epoch [8200/20000], Loss: 0.1032
    Epoch [8300/20000], Loss: 0.1022
    Epoch [8400/20000], Loss: 0.1013
    Epoch [8500/20000], Loss: 0.1004
    Epoch [8600/20000], Loss: 0.0995
    Epoch [8700/20000], Loss: 0.0986
    Epoch [8800/20000], Loss: 0.0978
    Epoch [8900/20000], Loss: 0.0969
    Epoch [9000/20000], Loss: 0.0961
    Epoch [9100/20000], Loss: 0.0954
    Epoch [9200/20000], Loss: 0.0946
    Epoch [9300/20000], Loss: 0.0939
    Epoch [9400/20000], Loss: 0.0932
    Epoch [9500/20000], Loss: 0.0925
    Epoch [9600/20000], Loss: 0.0918
    Epoch [9700/20000], Loss: 0.0911
    Epoch [9800/20000], Loss: 0.0905
    Epoch [9900/20000], Loss: 0.0898
    Epoch [10000/20000], Loss: 0.0892
    Epoch [10100/20000], Loss: 0.0886
    Epoch [10200/20000], Loss: 0.0880
    Epoch [10300/20000], Loss: 0.0875
    Epoch [10400/20000], Loss: 0.0869
    Epoch [10500/20000], Loss: 0.0864
    Epoch [10600/20000], Loss: 0.0858
    Epoch [10700/20000], Loss: 0.0853
    Epoch [10800/20000], Loss: 0.0848
    Epoch [10900/20000], Loss: 0.0843
    Epoch [11000/20000], Loss: 0.0838
    Epoch [11100/20000], Loss: 0.0834
    Epoch [11200/20000], Loss: 0.0829
    Epoch [11300/20000], Loss: 0.0825
    Epoch [11400/20000], Loss: 0.0820
    Epoch [11500/20000], Loss: 0.0816
    Epoch [11600/20000], Loss: 0.0812
    Epoch [11700/20000], Loss: 0.0808
    Epoch [11800/20000], Loss: 0.0803
    Epoch [11900/20000], Loss: 0.0799
    Epoch [12000/20000], Loss: 0.0796
    Epoch [12100/20000], Loss: 0.0792
    Epoch [12200/20000], Loss: 0.0788
    Epoch [12300/20000], Loss: 0.0784
    Epoch [12400/20000], Loss: 0.0781
    Epoch [12500/20000], Loss: 0.0777
    Epoch [12600/20000], Loss: 0.0774
    Epoch [12700/20000], Loss: 0.0771
    Epoch [12800/20000], Loss: 0.0767
    Epoch [12900/20000], Loss: 0.0764
    Epoch [13000/20000], Loss: 0.0761
    Epoch [13100/20000], Loss: 0.0758
    Epoch [13200/20000], Loss: 0.0755
    Epoch [13300/20000], Loss: 0.0752
    Epoch [13400/20000], Loss: 0.0749
    Epoch [13500/20000], Loss: 0.0746
    Epoch [13600/20000], Loss: 0.0743
    Epoch [13700/20000], Loss: 0.0740
    Epoch [13800/20000], Loss: 0.0737
    Epoch [13900/20000], Loss: 0.0735
    Epoch [14000/20000], Loss: 0.0732
    Epoch [14100/20000], Loss: 0.0729
    Epoch [14200/20000], Loss: 0.0727
    Epoch [14300/20000], Loss: 0.0724
    Epoch [14400/20000], Loss: 0.0722
    Epoch [14500/20000], Loss: 0.0719
    Epoch [14600/20000], Loss: 0.0717
    Epoch [14700/20000], Loss: 0.0715
    Epoch [14800/20000], Loss: 0.0712
    Epoch [14900/20000], Loss: 0.0710
    Epoch [15000/20000], Loss: 0.0708
    Epoch [15100/20000], Loss: 0.0706
    Epoch [15200/20000], Loss: 0.0704
    Epoch [15300/20000], Loss: 0.0701
    Epoch [15400/20000], Loss: 0.0699
    Epoch [15500/20000], Loss: 0.0697
    Epoch [15600/20000], Loss: 0.0695
    Epoch [15700/20000], Loss: 0.0693
    Epoch [15800/20000], Loss: 0.0691
    Epoch [15900/20000], Loss: 0.0689
    Epoch [16000/20000], Loss: 0.0687
    Epoch [16100/20000], Loss: 0.0685
    Epoch [16200/20000], Loss: 0.0684
    Epoch [16300/20000], Loss: 0.0682
    Epoch [16400/20000], Loss: 0.0680
    Epoch [16500/20000], Loss: 0.0678
    Epoch [16600/20000], Loss: 0.0676
    Epoch [16700/20000], Loss: 0.0675
    Epoch [16800/20000], Loss: 0.0673
    Epoch [16900/20000], Loss: 0.0671
    Epoch [17000/20000], Loss: 0.0670
    Epoch [17100/20000], Loss: 0.0668
    Epoch [17200/20000], Loss: 0.0666
    Epoch [17300/20000], Loss: 0.0665
    Epoch [17400/20000], Loss: 0.0663
    Epoch [17500/20000], Loss: 0.0662
    Epoch [17600/20000], Loss: 0.0660
    Epoch [17700/20000], Loss: 0.0659
    Epoch [17800/20000], Loss: 0.0657
    Epoch [17900/20000], Loss: 0.0656
    Epoch [18000/20000], Loss: 0.0654
    Epoch [18100/20000], Loss: 0.0653
    Epoch [18200/20000], Loss: 0.0652
    Epoch [18300/20000], Loss: 0.0650
    Epoch [18400/20000], Loss: 0.0649
    Epoch [18500/20000], Loss: 0.0647
    Epoch [18600/20000], Loss: 0.0646
    Epoch [18700/20000], Loss: 0.0645
    Epoch [18800/20000], Loss: 0.0643
    Epoch [18900/20000], Loss: 0.0642
    Epoch [19000/20000], Loss: 0.0641
    Epoch [19100/20000], Loss: 0.0640
    Epoch [19200/20000], Loss: 0.0638
    Epoch [19300/20000], Loss: 0.0637
    Epoch [19400/20000], Loss: 0.0636
    Epoch [19500/20000], Loss: 0.0635
    Epoch [19600/20000], Loss: 0.0634
    Epoch [19700/20000], Loss: 0.0632
    Epoch [19800/20000], Loss: 0.0631
    Epoch [19900/20000], Loss: 0.0630
    Epoch [20000/20000], Loss: 0.0629
    Training time: 16.85 seconds



    
![png](Day_37_files/Day_37_13_1.png)
    


## 3. 为什么 GPU 表现得更慢？

对于如此小的数据集和简单模型，GPU 往往比 CPU 慢，主要受三类固定开销影响：

1. **数据传输**：CPU 内存与 GPU 显存之间来回拷贝。
2. **核心（kernel）启动**：每个算子都要在 GPU 上启动一次核心程序。
3. **计算资源浪费**：批量小、计算量少，GPU 的并行能力发挥不出来。


### 3.1 数据传输细节

- 在 GPU 计算之前，输入张量、标签与模型参数都要从 RAM 复制到显存。
- `loss.item()` 每次都会把标量从 GPU 拷回 CPU，用于日志打印或可视化。
- 在 20000 个 epoch 的循环中，这些同步操作的总时间并不比实际计算少。


### 3.2 核心启动与批处理

- 每个前向或反向算子都会触发一次 kernel 启动，哪怕只是一个线性层。
- 当只有少量样本和极小的 batch 时，GPU 无法并行足够多的计算来摊平这些固定成本。
- 因此才会看到 “CPU 4.6 秒就跑完，而 GPU 却耗时 17 秒” 的现象。


### 3.3 何时使用 GPU

- 深度学习项目往往动辄几十分钟或数小时，此时 GPU 的高吞吐量能极大缩短训练时间。
- CPU 适合小任务，省去了数据跨芯片的传输。
- GPU 需要把数据、模型搬到显存，且频繁的 kernel 启动会放大额外成本。
- 当模型规模、数据集或 batch size 足够大时，GPU 才能发挥并行优势。


## 4. 减少额外开销的实践

针对上述瓶颈，最直接的方向是减少不必要的 CPU⇄GPU 往返。下面演示两个思路：

1. **彻底停止频繁记录**：不在循环中保存/打印 `loss.item()`，从根源上避免同步。
2. **降低记录频率**：例如改为每 200 个 epoch 才把损失值搬回来。既保留监控指标，也控制传输次数。



```python
# 思路1：完全不记录loss，纯粹观察终端输出
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 20000

start_time = time.time()

for epoch in range(num_epochs):
    outputs = model.forward(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Training time: {time.time() - start_time:.2f} seconds')

```

    Epoch [100/20000], Loss: 1.0762
    Epoch [200/20000], Loss: 1.0561
    Epoch [300/20000], Loss: 1.0299
    Epoch [400/20000], Loss: 0.9972
    Epoch [500/20000], Loss: 0.9581
    Epoch [600/20000], Loss: 0.9127
    Epoch [700/20000], Loss: 0.8619
    Epoch [800/20000], Loss: 0.8082
    Epoch [900/20000], Loss: 0.7560
    Epoch [1000/20000], Loss: 0.7061
    Epoch [1100/20000], Loss: 0.6604
    Epoch [1200/20000], Loss: 0.6188
    Epoch [1300/20000], Loss: 0.5814
    Epoch [1400/20000], Loss: 0.5478
    Epoch [1500/20000], Loss: 0.5178
    Epoch [1600/20000], Loss: 0.4908
    Epoch [1700/20000], Loss: 0.4664
    Epoch [1800/20000], Loss: 0.4443
    Epoch [1900/20000], Loss: 0.4242
    Epoch [2000/20000], Loss: 0.4061
    Epoch [2100/20000], Loss: 0.3894
    Epoch [2200/20000], Loss: 0.3740
    Epoch [2300/20000], Loss: 0.3595
    Epoch [2400/20000], Loss: 0.3459
    Epoch [2500/20000], Loss: 0.3332
    Epoch [2600/20000], Loss: 0.3211
    Epoch [2700/20000], Loss: 0.3097
    Epoch [2800/20000], Loss: 0.2988
    Epoch [2900/20000], Loss: 0.2885
    Epoch [3000/20000], Loss: 0.2787
    Epoch [3100/20000], Loss: 0.2695
    Epoch [3200/20000], Loss: 0.2607
    Epoch [3300/20000], Loss: 0.2524
    Epoch [3400/20000], Loss: 0.2445
    Epoch [3500/20000], Loss: 0.2370
    Epoch [3600/20000], Loss: 0.2299
    Epoch [3700/20000], Loss: 0.2232
    Epoch [3800/20000], Loss: 0.2168
    Epoch [3900/20000], Loss: 0.2108
    Epoch [4000/20000], Loss: 0.2051
    Epoch [4100/20000], Loss: 0.1997
    Epoch [4200/20000], Loss: 0.1945
    Epoch [4300/20000], Loss: 0.1896
    Epoch [4400/20000], Loss: 0.1850
    Epoch [4500/20000], Loss: 0.1805
    Epoch [4600/20000], Loss: 0.1763
    Epoch [4700/20000], Loss: 0.1723
    Epoch [4800/20000], Loss: 0.1685
    Epoch [4900/20000], Loss: 0.1648
    Epoch [5000/20000], Loss: 0.1613
    Epoch [5100/20000], Loss: 0.1580
    Epoch [5200/20000], Loss: 0.1548
    Epoch [5300/20000], Loss: 0.1518
    Epoch [5400/20000], Loss: 0.1489
    Epoch [5500/20000], Loss: 0.1461
    Epoch [5600/20000], Loss: 0.1434
    Epoch [5700/20000], Loss: 0.1409
    Epoch [5800/20000], Loss: 0.1384
    Epoch [5900/20000], Loss: 0.1361
    Epoch [6000/20000], Loss: 0.1338
    Epoch [6100/20000], Loss: 0.1317
    Epoch [6200/20000], Loss: 0.1296
    Epoch [6300/20000], Loss: 0.1276
    Epoch [6400/20000], Loss: 0.1257
    Epoch [6500/20000], Loss: 0.1239
    Epoch [6600/20000], Loss: 0.1221
    Epoch [6700/20000], Loss: 0.1204
    Epoch [6800/20000], Loss: 0.1187
    Epoch [6900/20000], Loss: 0.1171
    Epoch [7000/20000], Loss: 0.1156
    Epoch [7100/20000], Loss: 0.1141
    Epoch [7200/20000], Loss: 0.1127
    Epoch [7300/20000], Loss: 0.1113
    Epoch [7400/20000], Loss: 0.1100
    Epoch [7500/20000], Loss: 0.1087
    Epoch [7600/20000], Loss: 0.1075
    Epoch [7700/20000], Loss: 0.1063
    Epoch [7800/20000], Loss: 0.1051
    Epoch [7900/20000], Loss: 0.1040
    Epoch [8000/20000], Loss: 0.1029
    Epoch [8100/20000], Loss: 0.1018
    Epoch [8200/20000], Loss: 0.1008
    Epoch [8300/20000], Loss: 0.0998
    Epoch [8400/20000], Loss: 0.0989
    Epoch [8500/20000], Loss: 0.0980
    Epoch [8600/20000], Loss: 0.0971
    Epoch [8700/20000], Loss: 0.0962
    Epoch [8800/20000], Loss: 0.0953
    Epoch [8900/20000], Loss: 0.0945
    Epoch [9000/20000], Loss: 0.0937
    Epoch [9100/20000], Loss: 0.0929
    Epoch [9200/20000], Loss: 0.0922
    Epoch [9300/20000], Loss: 0.0914
    Epoch [9400/20000], Loss: 0.0907
    Epoch [9500/20000], Loss: 0.0900
    Epoch [9600/20000], Loss: 0.0894
    Epoch [9700/20000], Loss: 0.0887
    Epoch [9800/20000], Loss: 0.0881
    Epoch [9900/20000], Loss: 0.0874
    Epoch [10000/20000], Loss: 0.0868
    Epoch [10100/20000], Loss: 0.0863
    Epoch [10200/20000], Loss: 0.0857
    Epoch [10300/20000], Loss: 0.0851
    Epoch [10400/20000], Loss: 0.0846
    Epoch [10500/20000], Loss: 0.0840
    Epoch [10600/20000], Loss: 0.0835
    Epoch [10700/20000], Loss: 0.0830
    Epoch [10800/20000], Loss: 0.0825
    Epoch [10900/20000], Loss: 0.0820
    Epoch [11000/20000], Loss: 0.0816
    Epoch [11100/20000], Loss: 0.0811
    Epoch [11200/20000], Loss: 0.0807
    Epoch [11300/20000], Loss: 0.0802
    Epoch [11400/20000], Loss: 0.0798
    Epoch [11500/20000], Loss: 0.0794
    Epoch [11600/20000], Loss: 0.0790
    Epoch [11700/20000], Loss: 0.0786
    Epoch [11800/20000], Loss: 0.0782
    Epoch [11900/20000], Loss: 0.0778
    Epoch [12000/20000], Loss: 0.0774
    Epoch [12100/20000], Loss: 0.0770
    Epoch [12200/20000], Loss: 0.0767
    Epoch [12300/20000], Loss: 0.0763
    Epoch [12400/20000], Loss: 0.0760
    Epoch [12500/20000], Loss: 0.0756
    Epoch [12600/20000], Loss: 0.0753
    Epoch [12700/20000], Loss: 0.0750
    Epoch [12800/20000], Loss: 0.0747
    Epoch [12900/20000], Loss: 0.0744
    Epoch [13000/20000], Loss: 0.0740
    Epoch [13100/20000], Loss: 0.0737
    Epoch [13200/20000], Loss: 0.0735
    Epoch [13300/20000], Loss: 0.0732
    Epoch [13400/20000], Loss: 0.0729
    Epoch [13500/20000], Loss: 0.0726
    Epoch [13600/20000], Loss: 0.0723
    Epoch [13700/20000], Loss: 0.0721
    Epoch [13800/20000], Loss: 0.0718
    Epoch [13900/20000], Loss: 0.0715
    Epoch [14000/20000], Loss: 0.0713
    Epoch [14100/20000], Loss: 0.0710
    Epoch [14200/20000], Loss: 0.0708
    Epoch [14300/20000], Loss: 0.0706
    Epoch [14400/20000], Loss: 0.0703
    Epoch [14500/20000], Loss: 0.0701
    Epoch [14600/20000], Loss: 0.0699
    Epoch [14700/20000], Loss: 0.0697
    Epoch [14800/20000], Loss: 0.0694
    Epoch [14900/20000], Loss: 0.0692
    Epoch [15000/20000], Loss: 0.0690
    Epoch [15100/20000], Loss: 0.0688
    Epoch [15200/20000], Loss: 0.0686
    Epoch [15300/20000], Loss: 0.0684
    Epoch [15400/20000], Loss: 0.0682
    Epoch [15500/20000], Loss: 0.0680
    Epoch [15600/20000], Loss: 0.0678
    Epoch [15700/20000], Loss: 0.0676
    Epoch [15800/20000], Loss: 0.0674
    Epoch [15900/20000], Loss: 0.0672
    Epoch [16000/20000], Loss: 0.0671
    Epoch [16100/20000], Loss: 0.0669
    Epoch [16200/20000], Loss: 0.0667
    Epoch [16300/20000], Loss: 0.0665
    Epoch [16400/20000], Loss: 0.0664
    Epoch [16500/20000], Loss: 0.0662
    Epoch [16600/20000], Loss: 0.0660
    Epoch [16700/20000], Loss: 0.0659
    Epoch [16800/20000], Loss: 0.0657
    Epoch [16900/20000], Loss: 0.0656
    Epoch [17000/20000], Loss: 0.0654
    Epoch [17100/20000], Loss: 0.0653
    Epoch [17200/20000], Loss: 0.0651
    Epoch [17300/20000], Loss: 0.0650
    Epoch [17400/20000], Loss: 0.0648
    Epoch [17500/20000], Loss: 0.0647
    Epoch [17600/20000], Loss: 0.0645
    Epoch [17700/20000], Loss: 0.0644
    Epoch [17800/20000], Loss: 0.0642
    Epoch [17900/20000], Loss: 0.0641
    Epoch [18000/20000], Loss: 0.0640
    Epoch [18100/20000], Loss: 0.0638
    Epoch [18200/20000], Loss: 0.0637
    Epoch [18300/20000], Loss: 0.0636
    Epoch [18400/20000], Loss: 0.0635
    Epoch [18500/20000], Loss: 0.0633
    Epoch [18600/20000], Loss: 0.0632
    Epoch [18700/20000], Loss: 0.0631
    Epoch [18800/20000], Loss: 0.0630
    Epoch [18900/20000], Loss: 0.0628
    Epoch [19000/20000], Loss: 0.0627
    Epoch [19100/20000], Loss: 0.0626
    Epoch [19200/20000], Loss: 0.0625
    Epoch [19300/20000], Loss: 0.0624
    Epoch [19400/20000], Loss: 0.0623
    Epoch [19500/20000], Loss: 0.0621
    Epoch [19600/20000], Loss: 0.0620
    Epoch [19700/20000], Loss: 0.0619
    Epoch [19800/20000], Loss: 0.0618
    Epoch [19900/20000], Loss: 0.0617
    Epoch [20000/20000], Loss: 0.0616
    Training time: 5.21 seconds


实测下来，GPU 训练时间迅速下降到与 CPU 接近，说明大量时间确实耗在把标量搬回 CPU 上。



```python
# 思路2：降低记录频率，兼顾可视化与性能
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
num_epochs = 20000
losses = []

start_time = time.time()

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2000 == 0:
        losses.append(loss.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'Training time: {time.time() - start_time:.2f} seconds')

plt.plot(range(len(losses)), losses)
plt.xlabel('Checkpoint Index')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs (Sampled)')
plt.show()

```

    使用设备: cuda:0
    Epoch [2000/20000], Loss: 0.3663
    Epoch [4000/20000], Loss: 0.2058
    Epoch [6000/20000], Loss: 0.1371
    Epoch [8000/20000], Loss: 0.1054
    Epoch [10000/20000], Loss: 0.0886
    Epoch [12000/20000], Loss: 0.0787
    Epoch [14000/20000], Loss: 0.0723
    Epoch [16000/20000], Loss: 0.0679
    Epoch [18000/20000], Loss: 0.0646
    Epoch [20000/20000], Loss: 0.0622
    Training time: 13.08 seconds



    
![png](Day_37_files/Day_37_21_1.png)
    


### 4.1 记录频率与耗时的关系

以总 epoch=20000 为例，我在本地的测试如下，剩余时长=总时长−4.6s（纯计算时间）：

| 记录间隔（轮） | 记录次数（次） | 剩余时长（秒） |
|----------------|----------------|----------------|
| 100            | 200            | 10             |
| 200            | 100            | 9.35           |
| 1000           | 20             | 11.55          |
| 2000           | 10             | 8.5            |

可以看到记录次数越少，额外耗时会略有下降，但并非严格线性；真实项目中应结合监控需求取舍。


## 5. 认识 __call__ 方法

`nn.Linear`、`nn.Module` 的实例之所以可以被写成 `self.fc1(x)`，是因为它们实现了 `__call__`。在 Python 里，任何定义了 `__call__` 的对象都可以像函数一样被调用。



```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

```

在 `__init__` 中执行 `self.fc1 = nn.Linear(4, 10)` 时，`self.fc1` 变成了一个 `nn.Linear` 的实例。调用 `self.fc1(x)` 实际上会触发 `nn.Module.__call__`，该方法再去调用子类的 `forward`，从而完成前向传播。


### 5.1 无参数示例

`__call__` 可以在每次“函数式调用”时维护内部状态，非常适合封装可调用对象。



```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 输出: 1
print(counter())  # 输出: 2
print(counter.count)  # 输出: 2

```

    1
    2
    2


实例化只会发生一次（`counter = Counter()`），随后每次调用 `counter()` 都会触发 `__call__` 并更新内部的 `count`。


### 5.2 带参数示例

`__call__` 也能像普通函数一样接收参数；对象既能保存状态又能提供行为，非常适合需要“带记忆”的可调用单元。



```python
class Adder:
    def __call__(self, a, b):
        print("唱跳篮球rap")
        return a + b

adder = Adder()
print(adder(3, 5))  # 输出: 8

```

    唱跳篮球rap
    8

