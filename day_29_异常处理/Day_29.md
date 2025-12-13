# Day 29 异常处理

## 学习目标
> - 理解 Python 异常处理流程（try / except / else / finally）。
> - 知道最常见的报错类型以及触发方式。
> - 能够快速捕获单一或多种异常，并给出兜底策略。
> - 了解 finally 在资源管理、机器学习训练等场景的意义。

## 1. Python 异常处理机制
当运行时遇到非法状态，Python 会创建一个异常对象并沿调用栈逐层查找“谁来处理它”。如果没有任何代码捕获该异常，程序就会在输出 Traceback 后终止。合理地将可能失败的语句包裹在 try 块中，就能把失败控制在局部，并给出更优雅的退路。

### 1.1 try / except 语句骨架
- `try`: 放入可能抛出异常的代码。
- `except`: 捕获指定异常类型并给出补救。推荐显式写出异常类。
- `except Exception as exc`: 在调试阶段可以兜底，但要记录 `exc`，否则容易吞掉真正的 bug。把被捕获的异常对象赋值给变量 `exc`，方便在 `except` 块里查看具体信息，例如 `print(exc)`

```python
try:
    risky_operation()
except ValueError as exc:
    handle_value_error(exc)
except Exception as exc:
    log.error(exc)
    raise  # 或者给出默认返回值
```


### 1.2 try-except-else
`else` 块只有在 `try` 中的代码完全成功时才会执行，常用于“主要任务完成后才能执行的后续逻辑”，避免把新的潜在异常包进 `try` 而被误捕获。

### 1.3 try-except-else-finally
`finally` 无论前面发生什么都会运行，适合做资源清理（关闭文件、释放锁、同步日志等）。哪怕在 `try` 或 `except` 中提前 `return`，`finally` 也一定会执行。

## 2. 常见异常与定位技巧
遇到 Traceback 时，按“异常类型 → 文件/行号 → 触发语句 → 调用栈”顺序阅读，就能快速归因。以下列出常见异常及最容易忽略的细节。

### 2.1 SyntaxError —— 语法无法被解析
- 解释器在执行前就发现语法不合法，比如缺少冒号、括号、或者表达式中断。
- 解决方式：对照报错行重新检查缩进、符号是否成对。


```python
print("--- SyntaxError 示例，取消注释体验 ---")
# 示例 a: 缺少冒号
# def my_function()
#     print("Hello")

# 示例 b: 非法表达式
# x = 5 +
# print(x)

```

    --- SyntaxError 示例，取消注释体验 ---


### 2.2 NameError —— 名称未定义
- 使用了还未声明的变量/函数，或简单的拼写错误。
- 先检查作用域，其次确认变量名是否输入正确。


```python
print("--- NameError 示例，取消注释体验 ---")
# 示例 a: 变量未定义
# print(some_undefined_variable)

# 示例 b: 打错变量名
# print(my_lisst) # 变量名拼写错误

```

    --- NameError 示例，取消注释体验 ---


### 2.3 TypeError —— 类型不支持当前操作
- 常见于不同类型直接相加、将不可调用对象当作函数等。
- 关注操作两侧的真实类型，可借助 `type()` 调试。


```python
print("--- TypeError 示例，取消注释体验 ---")
# print("Age: " + 25) # 字符串和整数
# my_number = 10
# my_number() # 尝试像函数一样调用一个整数

```

    --- TypeError 示例，取消注释体验 ---


### 2.4 ValueError —— 值合法性问题
- 参数类型正确，但取值不符合要求，比如把无效字符串转成数字。
- 仔细查看报错信息中给出的“非法值”。


```python
print("--- ValueError 示例，取消注释体验 ---")
# my_string = "12.34.56"
# number = float(my_string) # '12.34.56' 不是一个有效的浮点数表示

```

    --- ValueError 示例，取消注释体验 ---


### 2.5 IndexError —— 序列下标越界
- 访问不存在的索引。对列表、元组、字符串都适用。
- 打印 `len(container)` 或者使用切片保护。


```python
print("--- IndexError 示例，取消注释体验 ---")
# animals = ("cat", "dog")
# print(animals[2])

```

    --- IndexError 示例，取消注释体验 ---


### 2.6 KeyError —— 字典缺少对应键
- 访问字典中不存在的 key。
- 可用 `dict.get(key)` 提供默认值，或在访问前 `if key in dict`.


```python
print("--- KeyError 示例，取消注释体验 ---")
# grades = {"math": 92, "science": 88}
# print(grades["history"])

```

    --- KeyError 示例，取消注释体验 ---


### 2.7 AttributeError —— 属性不存在
- 对象不具备该属性/方法，常见于把其他语言的习惯带到 Python。
- 通过 `dir(obj)` 或查阅文档确认真实 API。


```python
print("--- AttributeError 示例，取消注释体验 ---")
# 示例a
# a_string = "hello"
# print(a_string.length) # 字符串长度用 len(a_string)，不是 .length 属性

# 示例b
# import numpy as np
# arr = np.array([1,2,3])
# print(arr.non_existent_attribute)

```

    --- AttributeError 示例，取消注释体验 ---


### 2.8 ZeroDivisionError —— 除数为零
- 任何数字除以 0 都会触发。
- 在接收用户输入时务必先判断分母。


```python
print("--- ZeroDivisionError 示例，取消注释体验 ---")
# value = 10 / 0

```

    --- ZeroDivisionError 示例，取消注释体验 ---


### 2.9 FileNotFoundError —— 文件路径不存在
- 读取/写入路径错误，或文件尚未生成。
- 使用 `Path(path).resolve()` 看看最终路径，必要时加上 `exist_ok`。


```python
print("--- FileNotFoundError 示例，取消注释体验 ---")
# import pandas as pd
# pd.read_csv("missing.csv")

```

    --- FileNotFoundError 示例，取消注释体验 ---


### 2.10 ModuleNotFoundError —— 模块未安装或路径错误
- 第三方库未安装 / 虚拟环境没激活 / 自定义模块不在 `sys.path`。
- 依赖管理一定要靠 `requirements.txt` 或环境管理器记录。


```python
print("--- ModuleNotFoundError 示例，取消注释体验 ---")
# import imaginary_package

```

    --- ModuleNotFoundError 示例，取消注释体验 ---


## 3. 用 try-except 提升容错
将最容易失败的语句圈出来，在 except 中决定“回退策略/报警/给默认值”。不要等到模型训练了一小时才因为一个可控的输入异常崩掉。

### 3.1 未处理异常的样子
下面的代码一旦分母为 0，就会直接中断程序。


```python
print("--- 未捕获异常示例，可自行取消注释 ---")
# numerator = 10
# denominator = 0
# result = numerator / denominator
# print(result)

```

    --- 未捕获异常示例，可自行取消注释 ---


### 3.2 捕获单一异常
捕获特定异常类可以给用户更友好的提示，并在需要时提供默认值。


```python
print("--- try-except 捕获 ZeroDivisionError ---")
numerator = 10
denominator = 0

try:
    print("尝试进行除法运算...")
    result = numerator / denominator
except ZeroDivisionError as exc:
    print(f"无法计算：{exc}，采用默认值 0")
    result = 0
else:
    print(f"除法成功，结果为 {result}")
finally:
    print("无论成功失败都能走到这里")

print(f"最终结果: {result}")

```

    --- try-except 捕获 ZeroDivisionError ---
    尝试进行除法运算...
    无法计算：division by zero，采用默认值 0
    无论成功失败都能走到这里
    最终结果: 0


### 3.3 捕获不同类型的异常
可以在同一个 `try` 后面写多个 `except`，每个分支处理不同错误，必要时记录日志并将异常上抛。


```python
print("--- try-except 处理 TypeError ---")
x = "Total items: "
y = 5

try:
    print("尝试拼接字符串和数字...")
    message = x + y
except TypeError:
    message = x + str(y)
    print("捕获到类型不匹配，自动把数字转成字符串。")
else:
    print("直接拼接成功。")

print(message)

```

    --- try-except 处理 TypeError ---
    尝试拼接字符串和数字...
    捕获到类型不匹配，自动把数字转成字符串。
    Total items: 5


## 4. 多异常、else 与 finally 的综合案例
在生产代码中，我们往往需要：
1. 同时捕获不同类型的异常。
2. 仅在所有校验都通过时执行后续逻辑（`else`）。
3. 在任何情况下都释放资源或写入日志（`finally`）。


```python
print("--- 解析占比字符串，展示多异常 + else + finally ---")

def parse_ratio(text):
    print(f"尝试解析: {text}")
    file_handle = open("temp.log", "a", encoding="utf-8")
    try:
        numerator, denominator = text.split(":")
        ratio = int(numerator) / int(denominator)
    except ValueError as exc:
        file_handle.write(f"ValueError: {exc}")
        print("格式有误，请提供 '数字:数字' 格式。")
        return None
    except ZeroDivisionError:
        file_handle.write("ZeroDivisionError: 除数为 0")
        print("分母不能为 0。")
        return None
    else:
        print(f"解析成功，结果为 {ratio:.2f}")
        return ratio
    finally:
        file_handle.write("parse_ratio 调用结束")
        file_handle.close()
        print("日志文件已关闭。")

parse_ratio("45:15")
parse_ratio("42:0")
parse_ratio("bad-input")

```

    --- 解析占比字符串，展示多异常 + else + finally ---
    尝试解析: 45:15
    解析成功，结果为 3.00
    日志文件已关闭。
    尝试解析: 42:0
    分母不能为 0。
    日志文件已关闭。
    尝试解析: bad-input
    格式有误，请提供 '数字:数字' 格式。
    日志文件已关闭。


## 5. finally 在 ML / DL 项目中的常见场景
1. **写日志**：训练成功与否都要 flush/close 日志句柄，防止记录缺失。
2. **释放显存/锁资源**：特别是在多进程训练或需要显式释放 GPU handle 的框架中。
3. **回滚配置**：训练前临时修改了随机种子、环境变量，需要恢复全局状态。
4. **保存中间状态**：长时间训练被中断时，把 checkpoint 或统计信息落盘。
5. **关闭数据库/消息队列连接**：无论是否发生异常，连接都必须回收。
