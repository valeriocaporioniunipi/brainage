import unittest
from code.abspath import abs_path

class abs_path_checker(unittest.TestCase):

    def test_correct_path(self):

        absolute_path = abs_path("test_abspath.py", "tests")
        print(absolute_path)
