# DAY 33 类的装饰器

从"动态改造类"的角度再看装饰器：当功能要在多个类之间复用，却又不想回到每个类里修改源码时，类装饰器提供了一种即插即用的方案。


## 1. 类装饰器解决的痛点

- 同一批类要统一增加日志、统计、鉴权等横切逻辑。
- 已上线的类不想回头修改源码，只想在外部"套一层"。
- 需要把增强逻辑抽象出来，供多个类共享。

类装饰器的本质：**接收一个类，返回一个被修改过的类**，从而在不触碰类定义的情况下完成扩展。


## 2. 函数装饰器 vs. 类装饰器

| 维度 | 函数装饰器 | 类装饰器 |
| --- | --- | --- |
| 作用对象 | 函数 / 方法 | 类 |
| 传入参数 | `decorator(func)` | `decorator(cls)` |
| 返回值 | 通常是包裹后的函数（闭包） | 修改后的类（原类 / 新类） |
| 常见用途 | 日志、计时、权限验证等 | 为类批量增加属性、方法，或重写 `__init__` |
| 核心价值 | 不修改函数源码即可增强功能 | 不修改类定义即可扩展行为 |

理解这张对比表，你就能迅速判断需求是要改函数还是改类。


## 3. 设计类装饰器的步骤

1. 接收原始类，并备份需要被替换的方法（常见是 `__init__`）。
2. 写一个新方法，在其中加入增强逻辑，然后调用原方法。
3. 把新方法、额外属性绑定到类（如 `cls.log = log_message`）。
4. 返回修改好的类，供外部继续使用。

下面的示例展示了如何一次性为多个类补充日志功能。



```python
# 定义类装饰器：统一添加日志功能
def class_logger(cls):
    original_init = cls.__init__  # 备份原始构造函数

    def new_init(self, *args, **kwargs):
        print(f"[LOG] 实例化对象: {cls.__name__}")
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init  # 覆写 __init__

    def log_message(self, message):
        print(f"[LOG] {self.__class__.__name__}: {message}")

    cls.log = log_message  # 动态添加新方法
    return cls


@class_logger
class SimplePrinter:
    def __init__(self, name):
        self.name = name

    def print_text(self, text):
        print(f"{self.name}: {text}")


printer = SimplePrinter("Alice")
printer.print_text("Hello, World!")
printer.log("这是装饰器添加的日志方法")

```

    [LOG] 实例化对象: SimplePrinter
    Alice: Hello, World!
    [LOG] SimplePrinter: 这是装饰器添加的日志方法


## 4. 运行效果与解读

- 实例化时，新版 `__init__` 会先打印日志，再调用原构造函数。
- 原有的 `print_text` 行为保持不变，保证兼容。
- 每个被装饰的类都多了一个 `log` 方法，可以在任何地方调用。

因此，装饰器提供的是一种"批量加功能"的能力。


## 5. 两种方式定义方法

| 方式 | 写法 | 特点 |
| --- | --- | --- |
| 类内部定义 | 在 `class` 语句块中写 `def` | 语义直观，但类定义后不易扩展 |
| 外部赋值 | 先定义函数，再执行 `cls.fn = fn` | 运行期可随时添加 / 修改方法，装饰器常用 |

两种方式的本质都一样：把函数对象绑定到类属性上。外部赋值让我们无需打开类的源码就能增强它，这就是类装饰器的威力。


## 6. 语法糖

`@decorator` 是 `MyClass = decorator(MyClass)` 的简写。即使类早已定义，仍可以手动调用装饰器函数改写它——这意味着旧代码也能被安全地套上新能力。

装饰器的核心目标，是在不破坏原实现的前提下，动态、可控地扩展类或函数。

