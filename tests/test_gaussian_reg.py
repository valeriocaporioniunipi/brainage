import unittest
import numpy as np
from code.gaussian_reg import gaussian_reg

class BasicGaussianRegression(unittest.TestCase):

    def test_gaussian_reg(self):

        features = np.linspace(-100,100,10) #Initialization of a vector with 10 elements

        #Creation of a matrix with 10 rows and 11 columns
        for i in range(10):
            features = np.vstack([features, np.linspace(-100+i+1, 100-i-1, 10)])

        def gaussfunction(x, m, v):
            return (np.exp((-0.5)*(((x-m)/v)**2)))/(np.sqrt(2*np.pi*(v**2)))
            
        #Target array 
        target = np.linspace(-100,100,11)
        target = gaussfunction(target, 0, 50)
        print(target)
        print(type(target), np.shape(target))

        _, _, r2 = gaussian_reg(features, target, 5, False)
        self.assertAlmostEqual(r2, 1, places=2)
