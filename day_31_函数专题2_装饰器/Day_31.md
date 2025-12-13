# Day 31 · 装饰器专题

**学习目标**
> - 弄清楚执行顺序
> - 回顾纯函数版的计时代码，明确痛点
> - 理解装饰器是如何接收函数、返回新函数的
> - 学会用 `*args`/`**kwargs` 让装饰器可复用
> - 结合练习题，体会语法糖 `@decorator` 的真实含义

## 1. 问题动机：计时代码打乱主流程

我们要找出 2 到 9999 的所有质数（只能被 1 和自身整除的自然数），并记录运行耗时：

1. `is_prime` 判断一个数是否是质数。
2. `prime_nums` 遍历数值并调用 `is_prime`。
3. 同一个函数里还掺杂了计时代码，导致可读性下降。


```python
import time

def is_prime(num):
    if num < 2:
        return False
    if num == 2:
        return True
    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True

def prime_nums():
    start_time = time.time()
    primes = []
    for n in range(2, 10_000):
        if is_prime(n):
            primes.append(n)
    end_time = time.time()
    print(f"耗时 {end_time - start_time:.4f}s，找到 {len(primes)} 个质数")

prime_nums()

```

    耗时 0.0049s，找到 1229 个质数


### 小结
- 函数的主要职责是“找质数”。
- 计时代码虽然必要，但反复写在不同函数里就显得冗余。
- 如果能把计时代码抽出去复用，就能达到 DRY（Don't Repeat Yourself）。

## 2. 第一版装饰器：提取共同逻辑

装饰器本质是一个高阶函数：它接收函数 `func` 作为参数，返回一个新的函数 `wrapper`。

1. 在 `wrapper` 中处理跨函数的通用逻辑（这里是计时）。
2. `wrapper` 再去调用原函数。
3. `@display_time` 等价于 `prime_nums = display_time(prime_nums)`。


```python
import time

def display_time(func):
    def wrapper():
        start_time = time.time()
        func()
        end_time = time.time()
        print(f"函数 {func.__name__} 耗时 {end_time - start_time:.4f}s")
    return wrapper

@display_time
def prime_nums():
    primes = []
    for n in range(2, 10_000):
        if is_prime(n):
            primes.append(n)
    print(f"找到 {len(primes)} 个质数")

prime_nums()

```

    找到 1229 个质数
    函数 prime_nums 耗时 0.0041s


### 装饰器执行顺序回放
1. Python 先定义 `display_time`。
2. 定义 `prime_nums` 时，解释器立刻执行 `display_time(prime_nums)`。
3. `display_time` 返回的 `wrapper` 覆盖了原来的 `prime_nums` 引用。
4. 以后调用 `prime_nums()` 实际执行 `wrapper()`，再由 `wrapper` 调用原函数。

把 `prime_nums()` 加断点会发现调用栈从 `wrapper` → 原函数，帮助我们理清顺序。

## 3. 可复用装饰器：兼容任意参数和返回值

- 真正可复用的装饰器必须能接受被装饰函数的所有参数，因此使用 `*args`/`**kwargs`。
- 如果被装饰函数有返回值，装饰器必须把结果原样返回给调用者。


```python
import time

def display_time(func):
    """支持任意参数并返回原函数结果的计时器装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 耗时 {end_time - start_time:.4f}s")
        return result
    return wrapper

@display_time
def sum_range(start, end):
    total = 0
    for n in range(start, end):
        total += n
    return total

value = sum_range(1, 100_000)
print(f"区间和为 {value}")

```

    函数 sum_range 耗时 0.0018s
    区间和为 4999950000


返回值千万不能丢：否则装饰器会把 `None` 返回给调用方，导致上层逻辑出错。这也是调试装饰器时最常见的坑。

## 4. 今日作业

请用装饰器实现一个简单的日志功能，打印函数名与传入参数，再打印返回结果。


```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"开始执行函数 {func.__name__}，参数: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"执行完成，返回值: {result}")
        return result
    return wrapper

@logger
def multiply(a, b):
    return a * b

multiply(a=2, b=3)

```

    开始执行函数 multiply，参数: (), {'a': 2, 'b': 3}
    执行完成，返回值: 6





    6




```python
multiply(2, b=3)
```

    开始执行函数 multiply，参数: (2,), {'b': 3}
    执行完成，返回值: 6





    6




```python
multiply(a=2, 3)
```


      Cell In[21], line 1
        multiply(a=2, 3)
                       ^
    SyntaxError: positional argument follows keyword argument



最后一行会触发 `SyntaxError`：关键字参数必须写在所有位置参数之后。利用这个错误提示加深对参数顺序的理解。
