import p15_auto_gradient as deriv


def train(y, x, epoces, learning_rate):
    dy_dx = y.deriv(x)  #这里是否应该用y.deriv.deriv(x).? 这里认为y， x都已经是表达式了
    print(dy_dx)
    x0 = 1.0  #初始值
    for _ in range(epoces):
        x0 += - dy_dx.eval(**{x.name: x0}) * learning_rate  #凡是x.name变量都用x0取代？什么意思
    return x0


if __name__ == '__main__':
    x = deriv.Variable('x')

    for a in range(2,10+1):
        y = (x ** 2 - a) ** 2
        x_v = train(y, x, 2000, 0.01)
        print(a, a**0.5, x_v)

