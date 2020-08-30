import unittest
import numpy as np
from typing import List, Tuple

class Variable:
    def __init__(self, data: np.ndarray):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func) -> None:
        self.creator = func

    def backward(self) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, inputs: List[Variable]) -> List[Variable]:
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs

    def forward(self, xs: List[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError

    def backward(self, gys: float) -> Tuple[np.ndarray]:
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

class Add(Function):
    def forward(self, xs: List[np.ndarray]) -> Tuple[np.ndarray]:
        x0, x1 = xs
        y = x0 + x1
        return (y,)

# 関数の基底クラスのインスタンス化をまとめる
def square(x: Variable) -> Variable:
    return Square()(x)

def exp(x: Variable) -> Variable:
    return Exp()(x)

def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

def numerical_diff(f: Function, x: Variable, eps=1e-4) -> float:
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

