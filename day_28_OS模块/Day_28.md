# Day 28 OS模块

这里补充一下上一节对pipeline形参结构的解释：

管道工程中pipeline类接收的是一个包含多个小元组的 列表 作为输入。

可以这样理解这个结构：

1. 列表 []: 定义了步骤执行的先后顺序。Pipeline 会按照列表中的顺序依次处理数据。之所以用列表，是未来可以对这个列表进行修改。
2. 元组 (): 用于将每个操作的名称和执行该操作的对象捆绑在一起。

不用字典因为字典是无序的。

在简单的入门级项目中，可能只需要使用 pd.read_csv() 加载数据，而不需要直接操作文件路径。但是，当你开始处理图像数据集、自定义数据加载流程、保存和加载复杂的模型结构时，os 模块就会变得非常有用。


```python
import os

os.getcwd() # get current working directory 获取当前工作目录的绝对路径
```




    '/mnt/e/PythonPractice/day_28'




```python
os.listdir() # list directory 获取当前工作目录下的文件列表
```




    ['Day_28.ipynb', '元组和OS模块.ipynb']




```python
#    我们使用 r'' 原始字符串，这样就不需要写双反斜杠 \\，因为\会涉及到转义问题
path_a=r'/mnt/e/PythonPractice/day_28'
path_b='MyProjectData'
file='results.csv'

# 使用 os.path.join 将它们安全地拼接起来，os.path.join 会自动使用 Windows 的反斜杠 '\' 作为分隔符
file_path = os.path.join(path_a , path_b, file)

file_path
```




    '/mnt/e/PythonPractice/day_28/MyProjectData/results.csv'




```python
# os.environ 表现得像一个字典，包含所有的环境变量
# os.environ
```

## 目录树

`os.walk()` 可以把一个目录当成树来遍历，特别适合在云端或服务器上没有图形界面的场景下快速了解当前项目里有些什么文件。每次遍历都会返回一个包含三个元素的元组，分别告诉你：

1. `dirpath` —— 此刻走到的目录位置；
2. `dirnames` —— 这个目录里还有哪些子目录；
3. `filenames` —— 该目录下有哪些普通文件。

把返回结果想成“我在哪、有哪些文件夹、有哪些文件”这三句话，就很好理解了。


**示例目录结构（Markdown形式）**

```markdown
my_project/
├── data/
│   ├── processed/
│   └── raw/
│       └── data1.csv
├── src/
│   ├── models/
│   │   └── model_a.py
│   └── utils.py
├── main.py
└── README.md
```


当 `os.walk()` 运行在上面的目录时，每一轮都会给到我们一组 `(dirpath, dirnames, filenames)`：

- 第一轮在 `my_project`，会看到子目录 `['data', 'src']` 和文件 `['main.py', 'README.md']`；
- 第二轮进入 `data`，再分别看到 `processed`、`raw` 等层级；
- 以此类推直到整棵树被走完。

顺序是深度优先的，但在大多数实际场景里你只需要关注路径本身即可。



```python
import os

start_directory = os.getcwd()  # 这里用当前工作目录演示
print(f'--- 开始遍历目录: {start_directory} ---')

for dirpath, dirnames, filenames in os.walk(start_directory):
    print(f"\n当前目录: {dirpath}")
    print(f"子目录: {dirnames if dirnames else '无'}")
    print(f"文件: {filenames if filenames else '无'}")

```

    --- 开始遍历目录: /mnt/e/PythonPractice/day_28 ---
    
    当前目录: /mnt/e/PythonPractice/day_28
    子目录: 无
    文件: ['Day_28.ipynb', '元组和OS模块.ipynb']


习惯用 `os.walk()` 在代码里打印目录树，可以让你在调试数据加载、保存模型、查找配置文件时迅速定位问题。以后在远程训练任务里，你基本都会靠它来确认文件是否按预期生成。

