# DAY 46 TensorBoard 使用介绍


## 学习目标
- 理解 TensorBoard 的作用与数据流转方式
- 掌握 SummaryWriter 的核心用法（标量、图像、直方图、计算图）
- 通过 CIFAR-10 的 MLP / CNN 实战，生成可视化日志


## 一、TensorBoard 概览
TensorBoard 是深度学习训练过程的可视化面板，可用于：
- 观察 loss / acc 曲线，判断收敛或过拟合
- 查看模型结构图，快速确认网络连接
- 记录样本图像、参数分布，辅助排查训练异常

工作原理：训练时把指标、图像、直方图等写入日志文件（*.tfevents），TensorBoard 读取该目录并在网页展示。


## 二、准备环境与数据



```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

```

    Using device: cuda



```python
# 为了演示更快，这里截取少量样本；想要完整训练可去掉 Subset
def get_loaders(batch_size=128, limit_train=5000, limit_test=1000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    if limit_train:
        train_set = Subset(train_set, range(limit_train))
    if limit_test:
        test_set = Subset(test_set, range(limit_test))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'Train samples: {len(train_set)}, Test samples: {len(test_set)}')
    return train_loader, test_loader

train_loader, test_loader = get_loaders()
images, labels = next(iter(train_loader))
print('Sample batch shape:', images.shape)

```

    Train samples: 5000, Test samples: 1000
    Sample batch shape: torch.Size([128, 3, 32, 32])


## 三、创建 SummaryWriter 与基础可视化



```python
# 创建 writer，日志会自动追加编号避免覆盖
writer = SummaryWriter(log_dir='runs/day46_intro')

# 记录一组训练图像
img_grid = make_grid(images[:16], nrow=8, normalize=True, scale_each=True)
writer.add_image('TrainSamples', img_grid, global_step=0)
writer.flush()
print('Logged sample images to runs/day46_intro')

```

    Logged sample images to runs/day46_intro


### 记录模型结构（Graph）



```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)

mlp = SimpleMLP().to(device)
dummy_input = images[:1].to(device)
writer.add_graph(mlp, dummy_input)
writer.flush()
print('Logged MLP graph')

```

    Logged MLP graph


## 四、MLP 训练 + TensorBoard 日志



```python
def train_mlp(epochs=2, log_dir='runs/day46_mlp'):
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, images[:1].to(device))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Acc/train', correct / total, global_step)
            global_step += 1

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Acc', epoch_acc, epoch)
        for name, param in model.named_parameters():
            if 'weight' in name:
                writer.add_histogram(name, param, epoch)

        print(f'Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}')

    writer.close()
    return model

mlp_model = train_mlp()

```

    Epoch 1: loss=2.0121, acc=0.3234
    Epoch 2: loss=1.6385, acc=0.4348


## 五、CNN 训练 + TensorBoard 日志



```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_cnn(epochs=2, log_dir='runs/day46_cnn'):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, images[:1].to(device))

    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Acc/train', correct / total, global_step)
            global_step += 1

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        writer.add_scalar('Epoch/Loss', epoch_loss, epoch)
        writer.add_scalar('Epoch/Acc', epoch_acc, epoch)
        writer.add_histogram('features.conv1.weight', model.features[0].weight, epoch)
        writer.add_histogram('features.conv2.weight', model.features[3].weight, epoch)

        print(f'Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}')

    writer.close()
    return model

cnn_model = train_cnn()

```

    Epoch 1: loss=1.9511, acc=0.2924
    Epoch 2: loss=1.5537, acc=0.4464


## 六、启动 TensorBoard
训练完成后在项目根目录执行：



```python
# tensorboard --logdir runs
# 浏览器打开 http://localhost:6006

```

## 七、常见问题与建议
- 直方图记录频率不宜过高，可按 epoch 记录减少日志体积
- 图像可用于检查数据增强是否符合预期
- 若曲线剧烈抖动，优先检查学习率、数据预处理和 batch size

