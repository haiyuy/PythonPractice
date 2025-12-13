# Day 32 · 类的定义和方法


## 1. 认识类

- 类把属性与方法组织在一起，是生成对象的模板。
- 实例化：用类创建对象后，才能真正使用这些属性和方法。
- 通过类我们能够复用结构化的代码，比如一次性定义好“教师”应该具备的特征，然后为不同的教师创建实例。


## 2. 定义类的基本语法

一个最基本的类由四个要素组成：
1. `class` 关键字。
2. 合法的类名，通常使用大驼峰。
3. 末尾的冒号 `:`，用于声明后续缩进块。
4. 类体（至少有一行有效语句）。


### pass 占位与缩进

在尚未决定类体写什么内容之前，可以用 `pass` 进行占位，保证语法完整。后续再逐行替换为真正的实现即可。



```python
class ClassName:
    """演示一个空类结构，类名遵循大驼峰命名。"""
    pass  # 用 pass 占位，避免出现缩进错误

```

### pass 在其他代码块中的用法

只要语句后面要求出现缩进块（如 `if/for/try` 等），又暂时不想写逻辑，都可以用 `pass`。



```python
x = 10
if x > 5:
    pass  # 在条件语句中临时占位，后续可以填充真正的逻辑
else:
    print("x 小于等于 5")

```


```python
for i in range(3):
    pass  # 循环体暂时为空

```


```python
try:
    print("尝试执行一些操作")
except Exception:
    pass  # 捕获到异常时暂不处理
finally:
    pass  # finally 块同样需要有实体语句

```

    尝试执行一些操作


> 小结：Python 完全依赖缩进划分代码块，解释器看到冒号后就期待一个缩进块。`pass` 是最简洁的有效语句，有它就能暂时“占坑”，等想好实现后再修改。


## 3. 初始化方法 `__init__`

- 又称构造方法，实例化对象时会被自动调用。
- 必须至少接受 `self` 参数，用来绑定实例自身。
- 可以在内部创建并初始化属性，让每个对象带着自己的数据出生。



```python
class Teacher:
    def __init__(self):
        self.name = "Susan"
        self.subject = "English"
        self.age = 33

teacher = Teacher()
print(teacher.name)

```

    Susan



```python
class Teacher:
    def __init__(self, name, age):
        self.name = name  # 通过 self 挂载到实例上
        self.age = age
        self.subject = "English"

teacher = Teacher("Susan", 33)
print(teacher.name)
print(teacher.age)
print(teacher.subject)

```

    Susan
    33
    English


`self.xxx` 表示“属于当前实例的属性”。即便属性值来自外部参数，也必须通过 `self` 赋值，实例之间才会互不干扰。


## 4. 普通方法与构造方法的区别

| 对比项 | `__init__` | 普通方法 |
| --- | --- | --- |
| 调用方式 | 实例化时自动触发 | 需要手动调用 |
| 命名 | 固定为 `__init__` | 任意描述性名称 |
| 用途 | 设置初始状态 | 表达对象的行为 |
| 参数 | 第一个参数必须是 `self` | 同样必须包含 `self` |
| 返回值 | 默认 `None` | 可返回任意结果 |

思考顺序是：先通过 `__init__` 把对象创建好，再用普通方法描述它的动作。



```python
class Teacher:
    def __init__(self):
        self.name = "Susan"
        self.subject = "English"
        self.age = 33

    def teach_lesson(self):
        print("上课中")

    def criticize(self):
        print("批评人")

teacher = Teacher()
teacher.teach_lesson()
teacher.criticize()
print(teacher.name)

```

    上课中
    批评人
    Susan



```python
class Teacher:
    def __init__(self, name, subject, age):
        self.name = name
        self.subject = subject
        self.age = age

    def teach_lesson(self):
        print(f"{self.name} 正在教 {self.subject}")

    def criticize(self, student_name):
        print(f"{self.name} 正在批评 {student_name}")

teacher = Teacher("Susan", "English", 33)
teacher.teach_lesson()
teacher.criticize("Mike")

```

    Susan 正在教 English
    Susan 正在批评 Mike


## 5. 继承：让类更可复用

- 子类可以继承父类的属性与方法，减少重复代码。
- 通过重写方法（同名覆盖）来定制特殊行为。
- 子类还可以新增自己的字段与函数。



```python
class Teacher:
    def __init__(self, name, subject, age):
        self.name = name
        self.subject = subject
        self.age = age

    def teach_lesson(self):
        print(f"{self.name} 正在教 {self.subject}")

    def criticize(self, student_name):
        print(f"{self.name} 正在批评 {student_name}")

class MasterTeacher(Teacher):
    def __init__(self, name, subject, age, experience_years):
        super().__init__(name, subject, age)
        self.experience_years = experience_years

    def teach_lesson(self):
        print(f"{self.name}（特级教师）正在用高级方法教授 {self.subject}")

    def give_lecture(self, topic):
        print(f"{self.name} 正在举办 {topic} 讲座")

master = MasterTeacher("王教授", "数学", 45, 20)
master.teach_lesson()
master.criticize("李同学")
master.give_lecture("微积分")

```

    王教授（特级教师）正在用高级方法教授 数学
    王教授 正在批评 李同学
    王教授 正在举办 微积分 讲座


## 6. `super()` 的常见用法

`super()` 返回父类对象，让我们在子类中沿用父类已经写好的逻辑。除了构造方法，普通方法中也能调用 `super()` 保留父类行为，再叠加子类特性。



```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print(f"{self.name} 发出声音")

class Dog(Animal):
    def speak(self):
        super().speak()
        print("汪汪叫")

dog = Dog("旺财", 3)
dog.speak()

```

    旺财 发出声音
    汪汪叫


> 方法重写的两种姿势：① 完全覆盖父类实现；② 借助 `super()` 先执行父类逻辑，再拓展子类行为。灵活组合就能写出既简洁又清晰的面向对象代码。

