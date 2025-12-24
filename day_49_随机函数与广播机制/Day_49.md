# Day 49 随机函数与广播机制

本节目标

- 用随机函数快速得到需要的张量形状（不用真实数据也能推导维度）
- 在卷积 / 池化 / 全连接中跟踪每一步的输出 shape
- 掌握广播规则（加法与矩阵乘法），理解形状兼容与值的扩展方式

## 1. 随机张量的生成

在模型原型或维度推导时，随机张量能快速占位。重点掌握 `torch.randn`，其他随机函数在分布上略有差异，但用法类似。

### 1.1 `torch.randn`：标准正态分布

- 生成均值 0、标准差 1 的随机张量
- 形状通过位置参数指定：`torch.randn(*size)`
- 可用于权重初始化、占位输入等


```python
import torch

# 为了可重复，固定随机种子
torch.manual_seed(0)

# 0 维：标量
scalar = torch.randn(())
print(f'标量: {scalar}, 形状: {scalar.shape}')

# 1 维：向量
vector = torch.randn(5)
print(f'向量: {vector}, 形状: {vector.shape}')

# 2 维：矩阵
matrix = torch.randn(3, 4)
print(f'矩阵形状: {matrix.shape}')

# 3 维：典型图像通道格式 (C, H, W)
tensor_3d = torch.randn(3, 224, 224)
print(f'3 维张量形状: {tensor_3d.shape}')

# 4 维：批量图像 [batch, channel, height, width]
tensor_4d = torch.randn(2, 3, 224, 224)
print(f'4 维张量形状: {tensor_4d.shape}')
```

    标量: 1.5409960746765137, 形状: torch.Size([])
    向量: tensor([-0.2934, -2.1788,  0.5684, -1.0845, -1.3986]), 形状: torch.Size([5])
    矩阵形状: torch.Size([3, 4])
    3 维张量形状: torch.Size([3, 224, 224])
    4 维张量形状: torch.Size([2, 3, 224, 224])


### 1.2 其他常见随机函数

- `torch.rand`：均匀分布，范围 [0, 1)
- `torch.randint`：整数分布，指定 `low`、`high` 与 `size`
- `torch.normal`：自定义均值、标准差的正态分布（均值和 std 支持逐元素）


```python
# 均匀分布随机数
uniform = torch.rand(3, 2)
print(f'均匀分布: {uniform}, 形状: {uniform.shape}')

# 随机整数
ints = torch.randint(low=0, high=10, size=(3,))
print(f'随机整数: {ints}, 形状: {ints.shape}')

# 自定义正态分布（逐元素均值/方差）
mean = torch.tensor([0.0, 0.0])
std = torch.tensor([1.0, 2.0])
normal = torch.normal(mean, std)
print(f'正态分布: {normal}, 形状: {normal.shape}')
```

    均匀分布: tensor([[0.6617, 0.2065],
            [0.7485, 0.7621],
            [0.7163, 0.6838]]), 形状: torch.Size([3, 2])
    随机整数: tensor([2, 7, 8]), 形状: torch.Size([3])
    正态分布: tensor([-1.0245, -1.4550]), 形状: torch.Size([2])


## 2. 用随机输入测试网络输出尺寸

常见套路：用随机张量推一遍网络，层层打印 shape，就能在没有真实数据时确认维度是否匹配。卷积/池化的输出高宽公式：

\[ H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - kernel}{stride} \right\rfloor + 1 \]



```python
import torch.nn as nn

# 模拟一张 CIFAR-10 大小的图片：batch=1, channel=3, 高宽=32
input_tensor = torch.randn(1, 3, 32, 32)
print(f'输入尺寸: {input_tensor.shape}')

# 1) 卷积层：保持高宽不变 (padding=1, kernel=3, stride=1)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
conv_output = conv(input_tensor)
print(f'卷积后: {conv_output.shape}')

# 2) 最大池化：高宽减半 (kernel=2, stride=2)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
pool_output = pool(conv_output)
print(f'池化后: {pool_output.shape}')

# 3) 展平：保留 batch 维，其他展开
flattened = pool_output.view(pool_output.size(0), -1)
print(f'展平后: {flattened.shape}')

# 4) 全连接层：输入特征数 = 16*16*16
fc1 = nn.Linear(in_features=4096, out_features=128)
fc_output = fc1(flattened)
print(f'线性层后: {fc_output.shape}')

# 5) 分类层：输出 10 类
fc2 = nn.Linear(128, 10)
final_output = fc2(fc_output)
print(f'最终输出: {final_output.shape}')
```

    输入尺寸: torch.Size([1, 3, 32, 32])
    卷积后: torch.Size([1, 16, 32, 32])
    池化后: torch.Size([1, 16, 16, 16])
    展平后: torch.Size([1, 4096])
    线性层后: torch.Size([1, 128])
    最终输出: torch.Size([1, 10])


Softmax 常用于多分类输出，把 logits 变成概率分布。


```python
softmax = nn.Softmax(dim=1)  # 在类别维度上做归一化
class_probs = softmax(final_output)
print(f'Softmax 概率: {class_probs}')
print(f'概率总和: {class_probs.sum(dim=1)}')
```

    Softmax 概率: tensor([[0.1090, 0.1139, 0.0948, 0.0888, 0.0950, 0.0958, 0.1150, 0.1160, 0.1021,
             0.0696]], grad_fn=<SoftmaxBackward0>)
    概率总和: tensor([1.], grad_fn=<SumBackward1>)


## 3. 广播机制 (Broadcasting)

在元素级运算时，如果形状不一致但兼容，PyTorch 会自动扩展张量：
- 从右向左比较维度
- 维度相同或其中一个为 1 才能对齐；缺失维度视为 1
- 输出维度取每一位的最大值
- 扩展是逻辑上的，不复制真实数据（内存友好）

### 3.1 加法的广播案例
关注两件事：最终形状、扩展出的值。


```python
# 2D + 1D：行向量会被复制到每一行
a = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = torch.tensor([10, 20, 30])             # (3,) -> (1, 3) -> (2, 3)
print('二维 + 一维 ->', (a + b).shape)
print(a + b)

# 3x1 与 1x3：分别沿列和行扩展
a2 = torch.tensor([[10], [20], [30]])  # (3, 1)
b2 = torch.tensor([1, 2, 3])          # (3,) -> (1, 3)
print('列向量 + 行向量 ->', (a2 + b2).shape)
print(a2 + b2)

# 3D + 2D：对齐右侧两个维度，batch 维按 1 扩展
a3 = torch.tensor([[[1], [2]], [[3], [4]]])  # (2, 2, 1)
b3 = torch.tensor([[10, 20]])               # (1, 2) -> (1, 1, 2)
print('三维 + 二维 ->', (a3 + b3).shape)
print(a3 + b3)

# 标量与张量：标量被视为 shape=()，会扩展到目标形状
a4 = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
print('二维 + 标量 ->', (a4 + 10).shape)
print(a4 + 10)
```

    二维 + 一维 -> torch.Size([2, 3])
    tensor([[11, 22, 33],
            [14, 25, 36]])
    列向量 + 行向量 -> torch.Size([3, 3])
    tensor([[11, 12, 13],
            [21, 22, 23],
            [31, 32, 33]])
    三维 + 二维 -> torch.Size([2, 2, 2])
    tensor([[[11, 21],
             [12, 22]],
    
            [[13, 23],
             [14, 24]]])
    二维 + 标量 -> torch.Size([2, 2])
    tensor([[11, 12],
            [13, 14]])


### 3.2 矩阵乘法中的广播

`matmul`/`@` 既遵循广播规则，也要满足矩阵乘法的最后两维约束：`A[..., m, n] @ B[..., n, p] -> [..., m, p]`。


```python
# 批量矩阵 @ 单个矩阵：单个矩阵会在 batch 维上扩展
A = torch.randn(2, 3, 4)  # (2, 3, 4)
B = torch.randn(4, 5)     # (4, 5) -> (1, 4, 5) -> (2, 4, 5)
print('批量 @ 单个 ->', torch.matmul(A, B).shape)

# 小批量 @ 可广播批量
A2 = torch.randn(3, 2, 4)  # (3, 2, 4)
B2 = torch.randn(1, 4, 5)  # (1, 4, 5) -> (3, 4, 5)
print('部分广播 ->', torch.matmul(A2, B2).shape)

# 更高维：自动对齐 batch 维
A3 = torch.randn(2, 3, 4, 5)  # (2, 3, 4, 5)
B3 = torch.randn(5, 6)        # (5, 6) -> (1, 1, 5, 6)
print('高维广播 ->', torch.matmul(A3, B3).shape)
```

    批量 @ 单个 -> torch.Size([2, 3, 5])
    部分广播 -> torch.Size([3, 2, 5])
    高维广播 -> torch.Size([2, 3, 4, 6])

