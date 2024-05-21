import unittest
import numpy as np
from matplotlib import pyplot as plt
import sys
import os

# Ottieni il percorso assoluto della directory che contiene il file corrente
current_dir = os.path.dirname(__file__)

# Risali di una directory (per uscire da 'other_directory') e poi entra nella directory 'brainage'
brainage_dir = os.path.abspath(os.path.join(current_dir, '..', 'brainage'))

# Aggiungi 'brainage' al percorso di ricerca dei moduli
sys.path.insert(0, brainage_dir)

from brainage.gaussian_reg import gaussian_reg

class BasicGaussianRegression(unittest.TestCase):

    def test_gaussian_reg(self):

        def gaussfunction(x, m, v):
            return (np.exp((-0.5)*(((x-m)/v)**2)))/(np.sqrt(2*np.pi*(v**2)))

        # Generazione di dati di esempio
        def generate_perfect_gp_data(m, n):
            X = np.random.rand(m, n-1)
            coefficients = np.random.rand(n-1)
            linear_combination = X.dot(coefficients)
            y = gaussfunction(linear_combination, m/2, m/4)
            data = np.hstack((X, y.reshape(-1, 1)))
            return data

        m = 100  # Numero di righe
        n = 10    # Numero di colonne (n-1 features + 1 target)
        data = generate_perfect_gp_data(m, n)

        features = data[:, :-1]
        target = data[:, -1]

        _, _, r2 = gaussian_reg(features, target, 5, False)
        self.assertAlmostEqual(r2, 1, places=2)
