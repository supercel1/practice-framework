import sys, os
import unittest
import numpy as np

if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.core import add, mul, square, Variable
from dezero.config import no_grad
from dezero.utils import plot_dot_graph

class StepTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(z.data, np.array(13.0))
        self.assertTrue(isinstance(x.grad, Variable))

        a = Variable(np.array(2.0))
        b = add(a, a)
        b.backward()

        a.cleargrad()
        b = add(add(a, a), a)
        b.backward()

    def test_generation(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = add(square(a), square(a))
        y.backward()
        self.assertEqual(y.data, np.array(32.0))
        self.assertEqual(x.grad.data, np.array(64.0))

    def test_backward(self):
        x0 = Variable(np.array(1.0))
        x1 = Variable(np.array(1.0))
        t = add(x0, x1)
        y = add(x0, t)
        y.backward()
        self.assertEqual(x0.grad.data, np.array(2.0))
        self.assertEqual(x1.grad.data, np.array(1.0))

    def test_using_config(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)
            self.assertEqual(x.grad, None)

    def test_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))

        y = 2.0 * a
        z = a * 2.0
        self.assertEqual(y.data, np.array(6.0))
        self.assertEqual(z.data, np.array(6.0))

    def test_neg(self):
        a = Variable(np.array(2.0))
        b = -a
        self.assertEqual(b.data, np.array(-2.0))

    def test_sub(self):
        a = Variable(np.array(2.0))
        b = a - 3.0
        c = 3.0 - a

        self.assertEqual(b.data, np.array(-1.0))
        self.assertEqual(c.data, np.array(1.0))

    def test_div(self):
        a = Variable(np.array(4.0))
        b = Variable(np.array(2.0))
        c = a / b
        self.assertEqual(c.data, np.array(2.0))

        c.backward()
        self.assertEqual(a.grad.data, 1/2.0)
        self.assertEqual(b.grad.data, -4.0 / 2.0 ** 2)

    def test_pow(self):
        x = Variable(np.array(2.0))
        y = x ** 3
        self.assertEqual(y.data, np.array(8.0))

    def test_grad(self):
        def sphere(x, y):
            z = x ** 2 + y ** 2
            return z
        
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(x.grad.data, 2.0)
        self.assertEqual(y.grad.data, 2.0)

        def goldstein(x, y):
            z = (1 + (x + y + 1) ** 2 * (19 - 14*x + 3 * x ** 2 - 14*y + 6*x*y + 3*y**2)) * \
                (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
            return z

        a = Variable(np.array(1.0))
        b = Variable(np.array(1.0))
        c = goldstein(a, b)
        c.backward()
        self.assertEqual(a.grad.data, -5376.0)
        self.assertEqual(b.grad.data, 8064.0)

    def test_rosen(self):
        def rosenbrock(x0, x1):
            y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
            return y
        x0 = Variable(np.array(0.0))
        x1 = Variable(np.array(2.0))

        y = rosenbrock(x0, x1)
        y.backward()
        
