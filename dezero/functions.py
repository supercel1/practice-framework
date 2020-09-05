import numpy as np
from dezero.core import Variable, Function

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: float) -> float:
        x, = self.inputs
        return 2 * x * gy

def square(x: Variable) -> Variable:
    return Square()(x)

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: float) -> float:
        x, = self.inputs
        return np.exp(x) * gy

def exp(x: Variable) -> Variable:
    return Exp()(x)

class Sin(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x: Variable) -> Variable:
    return Sin()(x)

class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x: Variable):
    return Cos()(x)