import sys, os
import unittest
import numpy as np


# 環境変数 MY_MODULE_PATH を読み込み、sys.path へ設定
module_dir = os.getenv('MY_MODULE_PATH', default=os.getcwd())
sys.path.append(module_dir)

from steps import Add, Variable

class AddTest(unittest.TestCase):
    global module_dir

    def setUp(self):
        self.moduledir = os.path.join(module_dir, "steps")
        self.modulefilepath = os.path.join(self.moduledir, "step11.py")
        self.modulename = "steps.step11.py"
        self.Add = Add()

    def tearDown(self):
        del self.Add

    def test_forward(self):
        xs = [Variable(np.array(2)), Variable(np.array(3))]
        f = Add()
        ys = f(xs)
        y = ys[0]
        expected = np.array(5)
        self.assertEqual(y.data, expected)
