'''
TensorFlow的核心:
数据在不需要求导运算时，仅以表达式（字符串）的形式存在，仅当需要运算时，才转化为浮点数，进行前向及反向传播等运算。
本包仿照这种思路，实现了标量的前向传播表达式生成和自动反向传播求导
'''


import math
import numpy as np


class Exp:
    def eval(self, **env):  #env是字典参数
        pass

    def deriv(self, x):
        pass

    def __add__(self, other):
        other = to_exp(other)
        return Add(self, other)

    def __radd__(self, other):
        other = to_exp(other)
        return Add(other, self)

    def __sub__(self, other):
        other = to_exp(other)
        return Sub(self, other)

    def __rsub__(self, other):
        other = to_exp(other)
        return Sub(other, self)

    def __mul__(self, other):
        other = to_exp(other)
        if isinstance(other, Const) and other.value == 0:  #两个条件是不能反过来的，否则就会报错
            return 0
        return Mul(self, other)

    def __rmul__(self, other):
        other = to_exp(other)
        if isinstance(other, Const) and other.value == 0:
            return 0
        return Mul(other, self)

    def __truediv__(self, other):
        other = to_exp(other)
        return Truediv(self, other)

    def __rtruediv__(self, other):
        other = to_exp(other)
        if isinstance(other, Const) and other.value == 0:  #
            return 0
        return Truediv(other, self)

    def __pow__(self, power, modulo=None):
        return Pow(self, to_exp(power))

    def __rpow__(self, power, modulo=None):
        return Pow(to_exp(power), self)

    def __abs__(self):
        return  Abs(self)

def to_exp(value):
    if isinstance(value, Exp):
        return value
    if type(value) == str:
        return  Variable(value)
    if type(value) in (int, float):
        return Const(value)
    raise Exception('Can not convert to an Expression' % value)

class Abs(Exp):
    def __init__(self,value):
        self.value = value

    def __repr__(self):
        return '|%s|' % self.value

    def eval(self, **env):
        self.result = abs(self.value.eval(**env))
        return self. result

    def deriv(self, x):
        return (1 if self.result > 0 else -1) * self.value.deriv(x)


def log(exp):
    exp = to_exp(exp)
    return Log(exp)


class Add(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) + self.right.eval(**env)  #这里是浮点数加法

    def deriv(self, x):
        return self.left.deriv(x) + self.right.deriv(x)  #这里是对象加法

    def __repr__(self):
        return '(%s + %s)' % (self.left, self.right)

class Sub(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) - self.right.eval(**env)  #这里是浮点数加法

    def deriv(self, x):
        return self.left.deriv(x) - self.right.deriv(x)  #这里是对象加法

    def __repr__(self):
        return '(%s - %s)' % (self.left, self.right)

class Mul(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) * self.right.eval(**env)  #这里是浮点数加法

    def deriv(self, x):
        return self.left.deriv(x) * self.right + self.right.deriv(x) * self.left  #这里是对象加法

    def __repr__(self):
        return '(%s * %s)' % (self.left, self.right)

class Truediv(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        div = self.left.eval(**env)
        if div == 0:
            return 0
        der = self.right.eval(**env)
        return div / der

    def deriv(self, x):
        return (self.left.deriv(x) * self.right - self.right.deriv(x) * self.left) / self.right ** 2  #这里是对象加法

    def __repr__(self):
        return '(%s / %s)' % (self.left, self.right)

class Pow(Exp):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def eval(self, **env):
        return self.left.eval(**env) / self.right.eval(**env)  #这里是浮点数加法

    def deriv(self, x):
        if isinstance(self.right, Const):
            return self.right * self.left**(self.right-1) * self.left.deriv(x)
        result = self.right.deriv(x) * log(self.left) + self.right * self.left.deriv(x) / self.left
        return result * self

    def __repr__(self):
        return '%s ** %s' % (self.left, self.right)


#首先是常数，最基本的表达式
class Const(Exp):
    def __init__(self, value):
        self.value = value

    def eval(self, **env):
        return  self.value

    def deriv(self, x):
        return 0

    def __repr__(self):
        return str(self.value)

e = Const(math.e)
pi = Const(math.pi)

class Log(Exp):
    def __init__(self, value, base=e):
        self.value =  value
        self.base = base

    def eval(self, **env):
        return math.log(self.value.eval(**env), self.base.eval(**env))

    def deriv(self, x):
        if isinstance(self.base, Const):
            return self.value.deriv(x) / self.value / log(self.base)
        if isinstance(self.value, Const):
            return log(self.value) * self.base.deriv(x) / (self.base * log(self.base)**2)
        result = self.value.deriv(x) / self.value *log(self.base)
        result -= self.base.deriv(x) /self.base * log(self.value)
        # result /= log(self.base)**2
        return result / log(self.base)**2

    def __repr__(self):
        return 'log(%s, %s)' % (self.value, self.base)


class Variable(Exp):
    def __init__(self, name):
        self.name = name

    def eval(self, **env):
        if self.name in env:
            return env[self.name]
        raise Exception('Variable %s not found' % self.name)

    def deriv(self, x):
        if isinstance(x, Variable):
            x = x.name
        return 1 if self.name == x else 0  #变量对自己求导是1，对其他常数求导为0

    def __repr__(self):
        return self.name

def _test_deriv(y, x):
    dy_dx = y.deriv('x')  #不能用变量x，包会自动变x的
    print('-' * 100)
    print(y)
    print(dy_dx)
    print(dy_dx.eval(x=x))


if __name__ == '__main__':
    x = Variable('x')
    print(x.eval(x=3.14))

    y = 3+x
    dy_dx = y.deriv(x)
    print(y)
    print(dy_dx)

    print('*'*100)
    y = 4*x+3
    dy_dx = y.deriv(x)
    print(y)
    print(dy_dx)

    print('*'*100)
    y = 4 * x * x + 3 * x +5
    dy_dx = y.deriv(x)
    print(y)
    print(dy_dx)
    print(dy_dx.eval(x=1))

    print('*'*100)
    y= 1/x
    dy_dx = y.deriv(x)
    print(y)
    print(dy_dx)
    print(dy_dx.eval(x=0.5))


    _test_deriv(e ** (2*x), 0.5)


    y = abs(x**3)
    y.eval(x=0.5)
    _test_deriv(y, 0.5)
    #先正向计算一次，才能调用eval给abs里的result赋值，否则会报错