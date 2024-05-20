import unittest
import numpy as np
from code.linear_reg import linear_reg

class BasicLinearRegression(unittest.TestCase):

    def test_linear_reg(self):

        features = np.linspace(0,100,10) #Initialization of a vector with 10 elements

        #Creation of a matrix with 10 rows and 11 columns
        for i in range(10):
            features = np.vstack([features, np.linspace(i+1, 100-i-1, 10)])

        #Target array 
        target = np.array(np.linspace(30,60,11))

        _, _, r2 = linear_reg(features, target, 5, False)
        self.assertAlmostEqual(r2, 1, places=2)
