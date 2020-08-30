import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        self.input = input
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, gy: float):
        raise NotImplementedError

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: float) -> float:
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: float) -> float:
        x = self.input.data
        return np.exp(x) * gy

def numerical_diff(f: Function, x: Variable, eps=1e-4) -> float:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 動作確認

A = Square()
B = Exp()
C = Square()

# 順伝播
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)

# 逆伝播
c.grad = np.array(1.0)
b.grad = C.backward(c.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)