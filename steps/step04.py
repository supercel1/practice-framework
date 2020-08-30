import numpy as np

class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data

class Function:
    def __call__(self, input: Variable) -> Variable:
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

def numerical_diff(f: Function, x: Variable, eps=1e-4) -> float:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 動作確認

x = Variable(np.array(2.0))
f = Exp()
dy = numerical_diff(f, x)

print(dy)
print(type(dy))