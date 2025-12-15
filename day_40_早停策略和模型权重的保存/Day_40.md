[toc]

# Day 40 · 早停策略与模型权重的保存

- 示例使用 Iris 数据集 + 简单 MLP，默认在 CPU 运行，如使用 GPU 请修改 `device`。


## 1. 基线训练流程

复习数据预处理、模型定义、训练循环、损失可视化以及测试集评估的完整套路。



```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm库用于进度条显示
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 设置设备
device = "cpu"
print(f"使用设备: {device}")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型并移至GPU
model = MLP().to(device)

# 分类问题使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 20000  # 训练的轮数

# 用于存储每100个epoch的损失值和对应的epoch数
losses = []
epochs = []

start_time = time.time()  # 记录开始时间

# 创建tqdm进度条
with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)  # 隐式调用forward函数
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失值并更新进度条
        if (epoch + 1) % 200 == 0:
            losses.append(loss.item())
            epochs.append(epoch + 1)
            # 更新进度条的描述信息
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 每1000个epoch更新一次进度条
        if (epoch + 1) % 1000 == 0:
            pbar.update(1000)  # 更新进度条

    # 确保进度条达到100%
    if pbar.n < num_epochs:
        pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

time_all = time.time() - start_time  # 计算训练时间
print(f'Training time: {time_all:.2f} seconds')

# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.show()

# 在测试集上评估模型，此时model内部已经是训练好的参数了
# 评估模型
model.eval() # 设置模型为评估模式
with torch.no_grad(): # torch.no_grad()的作用是禁用梯度计算，可以提高模型推理速度
    outputs = model(X_test)  # 对测试数据进行前向传播，获得预测结果
    _, predicted = torch.max(outputs, 1) # torch.max(outputs, 1)返回每行的最大值和对应的索引
    #这个函数返回2个值，分别是最大值和对应索引，参数1是在第1维度（行）上找最大值，_ 是Python的约定，表示忽略这个返回值，所以这个写法是找到每一行最大值的下标
    # 此时outputs是一个tensor，p每一行是一个样本，每一行有3个值，分别是属于3个类别的概率，取最大值的下标就是预测的类别


    # predicted == y_test判断预测值和真实值是否相等，返回一个tensor，1表示相等，0表示不等，然后求和，再除以y_test.size(0)得到准确率
    # 因为这个时候数据是tensor，所以需要用item()方法将tensor转化为Python的标量
    # 之所以不用sklearn的accuracy_score函数，是因为这个函数是在CPU上运行的，需要将数据转移到CPU上，这样会慢一些
    # size(0)获取第0维的长度，即样本数量

    correct = (predicted == y_test).sum().item() # 计算预测正确的样本数
    accuracy = correct / y_test.size(0)
    print(f'测试集准确率: {accuracy * 100:.2f}%')
```

    使用设备: cpu


    训练进度: 100%|██████████| 20000/20000 [00:04<00:00, 4001.73epoch/s, Loss=0.0666]


    Training time: 5.00 seconds




![png](Day_40_files/Day_40_2_3.png)
    


    测试集准确率: 96.67%


## 2. 同步监控测试集

- 训练集 loss 持续下降 ≠ 模型泛化良好，测试集可能在增大，说明过拟合。
- 实战中最好同步记录测试集 loss/指标，并用一张图观察双方走势。

### 2.1 过拟合的典型特征
- 正常：训练/测试损失一起下降直至稳定。
- 过拟合：训练损失继续下降，而测试损失上升或震荡不再下降。



```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm库用于进度条显示
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 设置设备
device = "cpu"
print(f"使用设备: {device}")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型并移至GPU
model = MLP().to(device)

# 分类问题使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 20000  # 训练的轮数

# 用于存储每200个epoch的损失值和对应的epoch数
train_losses = [] # 存储训练集损失
test_losses = [] # 新增：存储测试集损失
epochs = []

start_time = time.time()  # 记录开始时间

# 创建tqdm进度条
with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)  # 隐式调用forward函数
        train_loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 记录损失值并更新进度条
        if (epoch + 1) % 200 == 0:
            # 计算测试集损失，新增代码
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            model.train()
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            epochs.append(epoch + 1)
            
            # 更新进度条的描述信息
            pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Test Loss': f'{test_loss.item():.4f}'})

        # 每1000个epoch更新一次进度条
        if (epoch + 1) % 1000 == 0:
            pbar.update(1000)  # 更新进度条

    # 确保进度条达到100%
    if pbar.n < num_epochs:
        pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

time_all = time.time() - start_time  # 计算训练时间
print(f'Training time: {time_all:.2f} seconds')

# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss') # 原始代码已有
plt.plot(epochs, test_losses, label='Test Loss')  # 新增：测试集损失曲线
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend() # 新增：显示图例
plt.grid(True)
plt.show()

# 在测试集上评估模型，此时model内部已经是训练好的参数了
# 评估模型
model.eval() # 设置模型为评估模式
with torch.no_grad(): # torch.no_grad()的作用是禁用梯度计算，可以提高模型推理速度
    outputs = model(X_test)  # 对测试数据进行前向传播，获得预测结果
    _, predicted = torch.max(outputs, 1) # torch.max(outputs, 1)返回每行的最大值和对应的索引
    correct = (predicted == y_test).sum().item() # 计算预测正确的样本数
    accuracy = correct / y_test.size(0)
    print(f'测试集准确率: {accuracy * 100:.2f}%')    
```

    使用设备: cpu


    训练进度: 100%|██████████| 20000/20000 [00:08<00:00, 2344.32epoch/s, Train Loss=0.0635, Test Loss=0.0599]


    Training time: 8.53 seconds




![png](Day_40_files/Day_40_4_3.png)
    


    测试集准确率: 96.67%


## 3. 模型权重的保存与加载

深度学习训练通常需要定期持久化模型或训练状态，常见颗粒度如下。

### 3.1 仅保存模型参数（推荐）
- 保存内容：`state_dict`（权重参数）。
- 特点：文件小、最常用，但加载前需重新定义与训练一致的模型结构。



```python
# 保存模型参数
torch.save(model.state_dict(), "model_weights.pth")
```

### 3.2 加载已保存的参数
- 步骤：先构造同一模型，再 `load_state_dict`。
- 适用：推理或继续训练，配合 `model.eval()` 进入推理模式。



```python
# 加载参数（需先定义模型结构）
model = MLP()  # 初始化与训练时相同的模型结构
model.load_state_dict(torch.load("model_weights.pth"))
# model.eval()  # 切换至推理模式（可选）
```




    <All keys matched successfully>



### 3.3 保存整个模型
- 保存内容：结构 + 参数。
- 优点：加载时无需重新定义类；缺点是文件大且依赖原始代码环境（自定义层可能报错）。



```python
# 保存整个模型
torch.save(model, "full_model.pth")

# 加载模型（无需提前定义类，但需确保环境一致）
model = torch.load("full_model.pth", weights_only=False)
model.eval()  # 切换至推理模式（可选）
```




    MLP(
      (fc1): Linear(in_features=4, out_features=10, bias=True)
      (relu): ReLU()
      (fc2): Linear(in_features=10, out_features=3, bias=True)
    )



### 3.4 保存训练状态（Checkpoint）
- 保存内容：模型参数、优化器状态、当前 epoch、loss 等，可用于断点续训。
- 适合：长时间训练或需要频繁中断的场景，也方便配合早停策略回退到最佳点。



```python
# # 保存训练状态
# checkpoint = {
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     "epoch": epoch,
#     "loss": best_loss,
# }
# torch.save(checkpoint, "checkpoint.pth")

# # 加载并续训
# model = MLP()
# optimizer = torch.optim.Adam(model.parameters())
# checkpoint = torch.load("checkpoint.pth")

# model.load_state_dict(checkpoint["model_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# start_epoch = checkpoint["epoch"] + 1  # 从下一轮开始训练
# best_loss = checkpoint["loss"]

# # 继续训练循环
# for epoch in range(start_epoch, num_epochs):
#     train(model, optimizer, ...)
```

## 4. 早停策略（Early Stopping）

- 目标：当验证集表现连续多次无提升时提前结束训练，避免对训练集过拟合。
- 核心：持续监控验证损失或指标，记录最佳结果并保存模型。

### 4.1 逻辑流程
1. 维护 `best_loss` 与 `counter`，每隔固定 epoch 评估一次验证集。
2. 若当前 loss 更低，则更新 `best_loss`、清零 `counter`、保存模型。
3. 否则 `counter += 1`，当 `counter >= patience` 时触发早停。



```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # 导入tqdm库用于进度条显示
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 设置GPU设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data  # 特征数据
y = iris.target  # 标签数据

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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型并移至GPU
model = MLP().to(device)

# 分类问题使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 使用随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 20000  # 训练的轮数

# 用于存储每200个epoch的损失值和对应的epoch数
train_losses = []  # 存储训练集损失
test_losses = []   # 存储测试集损失
epochs = []

# ===== 新增早停相关参数 =====
best_test_loss = float('inf')  # 记录最佳测试集损失
best_epoch = 0                 # 记录最佳epoch
patience = 50                # 早停耐心值（连续多少轮测试集损失未改善时停止训练）
counter = 0                    # 早停计数器
early_stopped = False          # 是否早停标志
# ==========================

start_time = time.time()  # 记录开始时间

# 创建tqdm进度条
with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(X_train)  # 隐式调用forward函数
        train_loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # 记录损失值并更新进度条
        if (epoch + 1) % 200 == 0:
            # 计算测试集损失
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            model.train()
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            epochs.append(epoch + 1)
            
            # 更新进度条的描述信息
            pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Test Loss': f'{test_loss.item():.4f}'})
            
            # ===== 新增早停逻辑 =====
            if test_loss.item() < best_test_loss: # 如果当前测试集损失小于最佳损失
                best_test_loss = test_loss.item() # 更新最佳损失
                best_epoch = epoch + 1 # 更新最佳epoch
                counter = 0 # 重置计数器
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停触发！在第{epoch+1}轮，测试集损失已有{patience}轮未改善。")
                    print(f"最佳测试集损失出现在第{best_epoch}轮，损失值为{best_test_loss:.4f}")
                    early_stopped = True
                    break  # 终止训练循环
            # ======================

        # 每1000个epoch更新一次进度条
        if (epoch + 1) % 1000 == 0:
            pbar.update(1000)  # 更新进度条

    # 确保进度条达到100%
    if pbar.n < num_epochs:
        pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

time_all = time.time() - start_time  # 计算训练时间
print(f'Training time: {time_all:.2f} seconds')

# ===== 新增：加载最佳模型用于最终评估 =====
if early_stopped:
    print(f"加载第{best_epoch}轮的最佳模型进行最终评估...")
    model.load_state_dict(torch.load('best_model.pth'))
# ================================

# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f'测试集准确率: {accuracy * 100:.2f}%')    
```

    使用设备: cuda:0


    训练进度: 100%|██████████| 20000/20000 [00:13<00:00, 1461.66epoch/s, Train Loss=0.0604, Test Loss=0.0509]


    Training time: 13.68 seconds




![png](Day_40_files/Day_40_14_3.png)
    


    测试集准确率: 96.67%


### 4.2 提示
- `patience` 过大时可能观察不到早停，适当调小可更快触发。
- 若训练结束未早停，说明测试集损失尚未出现连续恶化。
- 最终部署/复训建议加载保存的 checkpoint，而非最后一次参数。

