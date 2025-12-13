# Day 35 · 文件的规范拆分和写法


昨天我们讨论了模块导入路径，这一节把焦点放在"如何把一个大文件拆成一组职责明确的文件"。拆分的目标是为了让代码更清晰、更易维护、更易复用。


## 为什么要拆分项目文件

- 提升可读性：每个文件只承载一种职责，新人可以快速理解项目结构。
- 降低维护成本：定位 bug 或新增特性时，只需要触达相关的独立模块。
- 便于复用：通用逻辑（如数据加载、日志工具）抽离成独立文件或包，后续项目直接 import 即可。
- 配合团队协作：不同人可以同时迭代互不冲突的模块。


## 机器学习项目流程概览
一个标准的 ML 项目往往按照“数据→模型→产出”逐步推进。拆分文件时，只需把每个阶段映射到独立脚本，命名就自然确定。


### 阶段与命名示例
- **数据加载** → `load_data.py` / `data_loader.py`：统一入口读取文件、数据库、API 等原始数据。
- **探索与可视化** → `eda.py` / `visualization_utils.py`：初期放在 Notebook，成熟后凝练为绘图函数。
- **数据预处理** → `preprocess.py` / `data_cleaning.py`：负责缺失值、异常值、标准化、编码等。
- **特征工程** → `feature_engineering.py`：生成或筛选特征，常与业务逻辑强相关。
- **模型训练** → `model.py` + `train.py`：一个定义模型结构，另一个封装训练循环与超参。
- **模型评估** → `evaluate.py`：统一算指标、输出报告，保证评估口径一致。
- **模型推理** → `predict.py` / `inference.py`：加载权重，对线上或离线数据做预测。


## 推荐目录组织
典型的顶层目录可拆成“源代码 / 配置 / 实验 / 产出”四块，下方是一个最常见的骨架。


### 1. `src/` —— 核心源代码
- `src/data/`：与数据相关的模块
    - `load_data.py`：封装所有数据输入渠道。
    - `preprocess.py`：集中处理清洗、转换操作。
    - `feature_engineering.py`：组合、选择高价值特征。
- `src/models/`：模型生命周期
    - `model.py`：定义模型架构或算法。
    - `train.py`：训练脚本，保存 checkpoint。
    - `evaluate.py`：统一评估指标与报告逻辑。
    - `predict.py` / `inference.py`：对新数据做推理。
- `src/utils/`：通用工具
    - `io_utils.py`：文件/数据库读写。
    - `logging_utils.py`：日志配置。
    - `math_utils.py`、`plotting_utils.py`：自定义数值或可视化函数。


### 2. `config/` —— 配置集中管理
- `config.py` / `settings.py`：用 Python 表达的参数集合，便于 import。
- `config.yaml` / `config.json`：声明式配置，更容易被非 Python 服务读取。
- `.env`：敏感信息通过环境变量注入，搭配 `.gitignore` 避免泄露。


### 3. `notebooks/` 与 `experiments/` —— 探索记录
- `notebooks/initial_eda.ipynb`：保存探索性分析、验证图表。
- `experiments/model_experimentation.py`：批量尝试不同的模型或超参。
这部分通常是项目最先诞生的内容，后续把稳定产出迁入 `src/` 中。


### 4. 产出物目录
- `data/`
    - `raw/`：原始数据，仅追加不可覆盖。
    - `processed/`：预处理后的数据集，供训练/验证。
    - `interim/`（可选）：存中间结果，便于断点续跑。
- `models/`：统一存放 `.pkl`、`.h5`、`.joblib` 等模型文件。
- `reports/` 或 `output/`
    - `reports/evaluation_report.txt`：文本或 Markdown 报告。
    - `reports/visualizations/`：图像输出。
    - `output/logs/`：运行日志，方便排障。


### 5. 文件结构示例
```
credit_default_prediction/
│
├── data/                   # 数据文件夹
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
│
├── src/                   # 源代码
│   ├── __init__.py
│   ├── data/             # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   │
│   ├── models/           # 模型相关代码
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   └── visualization/    # 可视化相关代码
│       ├── __init__.py
│       └── plots.py
│
├── notebooks/            # Jupyter notebooks
│   └── model_development.ipynb
│
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明文档
```

## 拆分实施顺序（建议）
1. **按流程拆文件**：先把数据、模型、评估等阶段代码移入独立 `.py`。
2. **提炼工具模块**：将多处重复的 IO、日志、通用函数沉淀到 `utils/`。
3. **集中配置**：把硬编码路径、超参数和密钥挪到 `config/` 或 `.env`。
4. **隔离数据与模型产出**：建立 `data/` 与 `models/` 顶层目录，避免与源代码混杂。
完成以上四步后，项目结构基本达标，可根据需要继续细化。


## 注意事项
拆分文件只是手段，还需要在执行入口、编码和类型标注等细节上保持一致性。


### `if __name__ == "__main__"`
- 每个 `.py` 都是一个模块对象，只有被直接执行时 `__name__` 才等于 `"__main__"`。
- 将程序入口包裹在该判断里，可明确起点，且在被其他模块 import 时不会误执行顶层逻辑。
- 配合 `main()` 函数能更好地回收局部变量，也避免导入阶段初始化昂贵资源。


### 编码声明 `# -*- coding: utf-8 -*-`
- 该行必须出现在文件前两行，用于显式告知解释器采用 UTF-8（或指定编码）。
- Python 3 默认 UTF-8，但团队协作或老旧编辑器可能造成编码错乱，仍建议保留声明。
- 常见乱码除了编码声明问题，也可能是编辑器保存为其他编码，必要时手动转码。


#### 示例
下面展示含中文字符串的最小示例。



```python
# -*- coding: utf-8 -*-
msg = "你好，世界！"  # 中文字符串
print(msg)

```

### 类型注解
Python 仍是动态语言，但借助注解可以让 IDE更准确地提示，并配合 mypy 等工具做到静态检查。


#### 变量注解
格式为 `变量名: 类型`，可选地立即赋值。



```python
name: str = "Alice"
age: int = 30
height: float = 1.75
is_student: bool = False

```

#### 函数注解
在参数和返回值上标注类型，语法为 `def func(arg: Type) -> ReturnType:`。



```python
def add(a: int, b: int) -> int:
    return a + b

def greet(name: str) -> None:
    print(f"Hello, {name}")

```

#### 类注解
类属性可以仅声明不赋值，构造方法中再初始化，有助于 IDE 理解对象结构。



```python
class Rectangle:
    width: float
    height: float

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

```

整理好目录结构 + 入口逻辑 + 编码/类型细节，项目即可在团队中顺畅迭代，也方便未来复用。

