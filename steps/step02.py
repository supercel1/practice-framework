import numpy as np

class Variable:
    def __init__(self, data: numpy.ndarray):
        self.data = data

class Function:
    def __call__(self, input: Variable):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x: np.ndarray):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        return x ** 2

# 動作確認

x = Variable(np.array(10))
f = Square()
y = f(x)

print(type(np.array(2)))
print(type(y))
print(y.data)