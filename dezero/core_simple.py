import weakref
import numpy as np
from typing import List, Tuple, Union
from .config import Config

class Variable:
    __array_priority__ = 200
    
    def __init__(self, data: np.ndarray, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data
        self.grad = None
        self.name = name
        self.creator = None
        self.generation = 0

    def set_creator(self, func) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False) -> None:
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return "variable(" + p + ")"

class Function:
    def __call__(self, *inputs: Tuple[Variable]) -> List[Variable]:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs: List[np.ndarray]) -> Tuple[np.ndarray]:
        raise NotImplementedError

    def backward(self, gys: float) -> Tuple[np.ndarray]:
        raise NotImplementedError

class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** 2

    def backward(self, gy: float) -> float:
        x = self.inputs[0].data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def backward(self, gy: float) -> float:
        x = self.input.data
        return np.exp(x) * gy

class Neg(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return -x
    
    def backward(self, gy: float) -> float:
        return -gy

class Add(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 + x1
        return y

    def backward(self, gy: float) -> float:
        return gy, gy

class Sub(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 - x1
        return y

    def backward(self, gy: float) -> float:
        return gy, -gy

class Mul(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 * x1
        return y

    def backward(self, gy: float) -> float:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return x1 * gy, x0 * gy

class Div(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        y = x0 / x1
        return y

    def backward(self, gy: float) -> float:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (- x0 / x1 ** 2)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x ** self.c

    def backward(self, gy: float) -> float:
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

# 関数の基底クラスのインスタンス化をまとめる
def square(x: Variable) -> Variable:
    return Square()(x)

def exp(x: Variable) -> Variable:
    return Exp()(x)

def neg(x) -> Variable:
    return Neg()(x)

def add(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)

def sub(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Sub()(x1, x0)

def mul(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)

def div(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c) -> Variable:
    return Pow(c)(x)

# helper関数
def as_array(x) -> np.ndarray:
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj: Union[Variable, np.ndarray]) -> Variable:
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

# 演算子のオーバーロード
def setup_variable() -> None:
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__neg__ = neg
    Variable.__pow__ = pow
