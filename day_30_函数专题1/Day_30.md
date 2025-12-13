# DAY 30 — 函数专题 1


本节目标
> - 明确函数的基本写法和文档字符串
> - 理解形参与实参的区别与调用方式
> - 掌握 `return` 的作用及常见陷阱
> - 区分局部/全局变量的作用域
> - 熟悉常见的参数形式：位置、默认、`*args`、仅关键字、`**kwargs`


## 1. 函数的定义

最小模板：
```python

def function_name(parameter1, parameter2, ...):
    """函数的用途、参数、返回值"""
    # 函数体
    return value  # 可选
```
- `def`：开始定义函数。
- `function_name`：遵循小写+下划线的命名约定，表达清晰含义。
- 参数列表：可为空；括号不可省略。
- `Docstring`：用来描述用途/参数/返回值，便于 `help()` 和跳转查看。
- 函数体：缩进的代码块。
- `return`：返回结果，缺省时返回 `None`；`return` 后的语句不会执行。



```python
# 定义一个简单的问候函数（无参、无返回）
def greet():
    """打印一句问候语。"""
    message = "大家好！欢迎学习 Python 函数。"
    print(message)


greet()

```

    大家好！欢迎学习 Python 函数。


Docstring 快速查看
- 在 Notebook 中可用 `函数名.__doc__` 或 `help(函数名)` 查看。
- 在 IDE/编辑器中，按住 Ctrl/Command 点击函数名通常也能跳转到定义处。



```python
print(greet.__doc__)

```

    打印一句问候语。


## 2. 带参数的函数：形参与实参
- 形参（parameter）：定义时的占位符，如 `name`。
- 实参（argument）：调用时传入的实际值，如 `"张三"`。
- 最好使用关键字参数提高可读性，尤其参数较多时。



```python
# 定义一个带一个参数的问候函数
def greet_person(name):
    """根据给定的名字打印问候语。

    Args:
        name (str): 要问候的人的名字。
    """
    message = f"你好, {name}! 很高兴认识你。"
    print(message)


greet_person("张三")  # 实参传递给形参 name

```

    你好, 张三! 很高兴认识你。



```python
# 定义一个带多个参数的函数 (例如在机器学习中计算两个特征的和)
def add_features(feature1, feature2):
    """计算两个数值特征的和。

    Args:
        feature1 (float or int): 第一个特征值。
        feature2 (float or int): 第二个特征值。
    """
    total = feature1 + feature2
    print(f"{feature1} + {feature2} = {total}")


add_features(10, 25)

```

    10 + 25 = 35


提示：少传或错传参数会触发 `TypeError`，阅读报错信息能快速定位问题。


## 3. 返回值
- 函数可以返回任意对象（数字、字符串、列表、字典等）。
- 没有 `return` 或 `return` 后面为空时，默认返回 `None`。
- `return` 后的代码不会执行。



```python
# 计算和并返回结果
def calculate_sum(a, b):
    """计算两个数的和并返回结果。

    Args:
        a (float or int): 第一个数。
        b (float or int): 第二个数。

    Returns:
        float or int: 两个数的和。
    """
    result = a + b
    return result
    print("这行代码不会被执行")


calculate_sum(2, 3)

```




    5



返回容器类型示例：常见于数据预处理。



```python
# 函数可以返回列表、字典等容器类型
def preprocess_data(raw_data_points):
    """模拟数据预处理：将所有数据点乘以 2。"""
    processed = []
    for point in raw_data_points:
        processed.append(point * 2)
    return processed


data = [1, 2, 3, 4, 5]
processed_data = preprocess_data(data)

print(f"原始数据: {data}")
print(f"预处理后数据: {processed_data}")

```

    原始数据: [1, 2, 3, 4, 5]
    预处理后数据: [2, 4, 6, 8, 10]


## 4. 变量作用域
- 局部变量：在函数内部定义，函数结束后销毁。
- 全局变量：在函数外定义，函数内可读取；若要修改需 `global` 声明（初学阶段少用）。



```python
print("--- 变量作用域示例 ---")
global_var = "我是一个全局变量"


def scope_test():
    local_var = "我是一个局部变量"
    print(f"在函数内部，可以看到局部变量: '{local_var}'")
    print(f"在函数内部，也可以看到全局变量: '{global_var}'")
    # global_var = "尝试在函数内修改全局变量"  # 没有 global 声明时，这里会创建同名局部变量


scope_test()
print()
print(f"在函数外部，可以看到全局变量: '{global_var}'")
# print(local_var)  # NameError: 局部变量在函数外不可见

```

    --- 变量作用域示例 ---
    在函数内部，可以看到局部变量: '我是一个局部变量'
    在函数内部，也可以看到全局变量: '我是一个全局变量'
    
    在函数外部，可以看到全局变量: '我是一个全局变量'


## 5. 常见参数形式与推荐顺序
定义时的顺序通常遵循：
`必需位置参数 -> 默认参数 -> *args -> 仅关键字参数(常带默认) -> **kwargs`

- 位置参数：按顺序匹配。
- 默认参数：提供默认值，调用时可省略。
- `*args`：收集多余的位置参数为元组。
- 仅关键字参数：必须以 `key=value` 形式传入。
- `**kwargs`：收集多余的关键字参数为字典。

示例：大量参数时，用关键字方式可读性更高：
```python
plot_data(data, x_col, y_col, "blue", "-", True, False)
plot_data(data=my_data, x_column='time', y_column='value',
          color='blue', linestyle='-', show_grid=True, use_log_scale=False)
```



```python
# 位置参数 + 关键字参数

def describe_pet(animal_type, pet_name):
    """显示宠物的信息。"""
    print(f"我有一只 {animal_type}.")
    print(f"我的 {animal_type} 的名字叫 {pet_name.title()}.")


describe_pet("猫", "咪咪")  # 位置参数

describe_pet(animal_type="狗", pet_name="旺财")  # 关键字参数，顺序无关

```

    我有一只 猫.
    我的 猫 的名字叫 咪咪.
    我有一只 狗.
    我的 狗 的名字叫 旺财.


默认参数：带默认值的参数必须放在没有默认值的参数之后。



```python
# 带默认值的参数
def describe_pet_default(pet_name, animal_type="狗"):
    """显示宠物的信息，动物类型默认为狗。"""
    print(f"我有一只 {animal_type}.")
    print(f"我的 {animal_type} 的名字叫 {pet_name.title()}.")


describe_pet_default(pet_name="小黑")  # animal_type 使用默认值 "狗"
describe_pet_default(pet_name="雪球", animal_type="仓鼠")  # 覆盖默认值

```

    我有一只 狗.
    我的 狗 的名字叫 小黑.
    我有一只 仓鼠.
    我的 仓鼠 的名字叫 雪球.


`*args`：收集多余的位置参数为元组。



```python
# *args 示例：比萨配料

def make_pizza(size, *toppings):
    """概述要制作的比萨。"""
    print(f"制作一个 {size} 寸的比萨，配料如下:")
    if toppings:
        for topping in toppings:
            print(f"- {topping}")
    else:
        print("- 原味 (无额外配料)")


make_pizza(12, "蘑菇")
make_pizza(16, "香肠", "青椒", "洋葱")
make_pizza(9)

```

    制作一个 12 寸的比萨，配料如下:
    - 蘑菇
    制作一个 16 寸的比萨，配料如下:
    - 香肠
    - 青椒
    - 洋葱
    制作一个 9 寸的比萨，配料如下:
    - 原味 (无额外配料)


`**kwargs`：收集多余的关键字参数为字典。



```python
# **kwargs 示例：构建用户档案

def build_profile(first_name, last_name, **user_info):
    """创建一个包含用户所有信息的字典。"""
    profile = {"first_name": first_name, "last_name": last_name}
    for key, value in user_info.items():
        profile[key] = value
    return profile


user_profile = build_profile(
    '爱因斯坦', '阿尔伯特', location='普林斯顿', field='物理学', hobby='小提琴'
)
print(f"用户信息: {user_profile}")

```

    用户信息: {'first_name': '爱因斯坦', 'last_name': '阿尔伯特', 'location': '普林斯顿', 'field': '物理学', 'hobby': '小提琴'}


`*args` + `**kwargs` + 仅关键字参数：注意顺序，调用时更灵活。



```python
# 组合示例：混合使用 *args / **kwargs / 仅关键字参数
def process_data(id_num, name, *tags, status="pending", **details):
    print(f"ID: {id_num}")
    print(f"Name: {name}")
    print(f"Tags (*args): {tags}")
    print(f"Status: {status}")
    print(f"Details (**kwargs): {details}")
    print("-" * 20)


process_data(101, "Alice", "vip", "new_user", location="USA", age=30)
process_data(102, "Bob", status="active", department="Sales")
process_data(103, "Charlie", "admin")
process_data(name="David", id_num=104, profession="Engineer")

```

    ID: 101
    Name: Alice
    Tags (*args): ('vip', 'new_user')
    Status: pending
    Details (**kwargs): {'location': 'USA', 'age': 30}
    --------------------
    ID: 102
    Name: Bob
    Tags (*args): ()
    Status: active
    Details (**kwargs): {'department': 'Sales'}
    --------------------
    ID: 103
    Name: Charlie
    Tags (*args): ('admin',)
    Status: pending
    Details (**kwargs): {}
    --------------------
    ID: 104
    Name: David
    Tags (*args): ()
    Status: pending
    Details (**kwargs): {'profession': 'Engineer'}
    --------------------


小结与建议
- 优先写清晰的 Docstring，方便自查与协作。
- 需要返回结果时务必使用 `return`；只打印不等于返回。
- 参数较多时，优先使用关键字调用，减少位置错误。
- 谨慎修改全局变量；多数场景用参数和返回值传递数据即可。

