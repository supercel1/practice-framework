import sys, os
import unittest
import numpy as np


# 環境変数 MY_MODULE_PATH を読み込み、sys.path へ設定
module_dir = os.getenv('MY_MODULE_PATH', default=os.getcwd())
sys.path.append(module_dir)

from steps import add, square, Variable

class StepTest(unittest.TestCase):
    global module_dir

    def setUp(self):
        self.moduledir = os.path.join(module_dir, "steps")
        self.modulefilepath = os.path.join(self.moduledir, "core_simple.py")
        self.modulename = "steps.core_simple.py"

    def test_forward(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))
        
        z = add(square(x), square(y))
        z.backward()
        self.assertEqual(z.data, np.array(13.0))
        self.assertEqual(x.grad, 4.0)
        self.assertEqual(y.grad, 6.0)

        a = Variable(np.array(2.0))
        b = add(a, a)
        b.backward()
        self.assertEqual(a.grad, 2.0)

        a.cleargrad()
        b = add(add(a, a), a)
        b.backward()
        self.assertEqual(a.grad, 3.0)
