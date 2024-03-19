# lsh_recommender/__init__.py
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad as integrate
    

class OptimalBR:
    def false_positive_probability(threshold, b, r):
        _prob = lambda t: 1 - (1 - t ** float(r)) ** float(b)
        a, err = integrate(_prob, 0.0, threshold)
        return a

    def false_negative_probability(threshold, b, r):
        _prob = lambda t: (1 - t ** float(r)) ** float(b)
        a, err = integrate(_prob, threshold, 1.0)
        return a

    def error_function(params, n):
        b, r = params
        if b * r > 128:
            return 1000  # Penalize if b * r > 128
        threshold = 0.8  # Assuming threshold for similarity
        error = OptimalBR.false_positive_probability(threshold, b, r) * 0.2 + OptimalBR.false_negative_probability(threshold, b, r) * 0.8
        return error

    def compute_optimal_br(n):
        initial_guess = [1, 1]  # Initial guess for b, r
        bounds = [(1, 128), (1, 128)]  # Bounds for b, r

        # Minimize the error function
        result = minimize(OptimalBR.error_function, initial_guess, bounds=bounds, args=(n,))
        b_optimal, r_optimal = result.x
        return b_optimal, r_optimal