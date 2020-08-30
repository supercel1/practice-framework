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

class Exp(Function):
    def forward(self, x: numpy.ndarray):
        return np.exp(x)


