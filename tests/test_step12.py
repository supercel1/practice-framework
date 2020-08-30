import sys, os
import unittest
import numpy as np


# 環境変数 MY_MODULE_PATH を読み込み、sys.path へ設定
module_dir = os.getenv('MY_MODULE_PATH', default=os.getcwd())
sys.path.append(module_dir)

from steps import add, Variable

class AddTest(unittest.TestCase):
    global module_dir

    def setUp(self):
        self.moduledir = os.path.join(module_dir, "steps")
        self.modulefilepath = os.path.join(self.moduledir, "core_simple.py")
        self.modulename = "steps.core_simple.py"

    def test_forward(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        y = add(x0, x1)
        expected = np.array(5)
        print(y.data, expected)
        self.assertEqual(y.data, expected)
