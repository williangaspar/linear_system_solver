import unittest
import numpy as np
from fractions import Fraction

from linear_sistem_solver import auto_add, auto_multiply, auto_swap, check_is_pivot, reverse_matrix, solve_linear_system, swap_row

class TestSwapRow(unittest.TestCase):
    def test_swap_row_2x2(self):
        matrix = np.array([[1, 2], [3, 4]])
        expected = np.array([[3, 4], [1, 2]])
        self.assertTrue((swap_row(matrix, 0, 1) == expected).all())

    def test_swap_row_3x3(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]])
        self.assertTrue((swap_row(matrix, 0, 1) == expected).all())

    def test_swap_row_3x3_2(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6]])
        self.assertTrue((swap_row(matrix, 1, 2) == expected).all())

class TestCheckIsPivot(unittest.TestCase):
    def test_check_is_pivot_2x2(self):
        row = np.array([1, 2])
        self.assertTrue(check_is_pivot(row, 0))

    def test_check_is_pivot_2x2_2(self):
        row = np.array([0, 2])
        self.assertFalse(check_is_pivot(row, 0))
    
    def test_check_is_pivot_3x3(self):
        row = np.array([0, 0, 3])
        self.assertTrue(check_is_pivot(row, 2))

    def test_check_is_pivot_3x3_2(self):
        row = np.array([0, 0, 3])
        self.assertFalse(check_is_pivot(row, 1))

class TestAutoSwap(unittest.TestCase):
    def test_auto_swap_2x2(self):
        matrix = np.array([[0, 2], [1, 0]])
        expected = np.array([[1, 0], [0, 2]])
        self.assertTrue((auto_swap(matrix) == expected).all())

    def test_auto_swap_3x3(self):
        matrix = np.array([[0, 2, 3], [1, 0, 3], [1, 2, 0]])
        expected = np.array([[1, 0, 3], [0, 2, 3], [1, 2, 0]])
        self.assertTrue((auto_swap(matrix) == expected).all())

    def test_auto_swap_3x3_2(self):
        matrix = np.array([[0, 2, 3], [1, 0, 3], [1, 2, 0]])
        expected = np.array([[1, 0, 3], [0, 2, 3], [1, 2, 0]])
        self.assertTrue((auto_swap(matrix) == expected).all())
    
    def test_auto_swap_3x3_3(self):
        matrix = np.array([[0, 2, 3], [0, 0, 3], [1, 2, 0]])
        expected = np.array([[1, 2, 0], [0, 2, 3], [0, 0, 3]])
        self.assertTrue((auto_swap(matrix) == expected).all())
    
    def test_auto_swap_3x3_4(self):
        matrix = np.array([[0, 2, 3], [0, 0, 3], [0, 2, 0]])
        expected = np.array([[0, 2, 0], [0, 2, 3], [0, 0, 3]])
        self.assertTrue((auto_swap(matrix) == expected).all())

class TestAutoMultiply(unittest.TestCase):
    def test_auto_multiply_3x3(self):
        matrix = np.array([[3, 3, 3], [0, 2, 2], [1, 1, 1]])
        expected = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]])
        self.assertTrue((auto_multiply(matrix) == expected).all())
    
    def test_auto_multiply_3x4(self):
        matrix = np.array([[3, 3, 3, 3], [0, 2, 2, 2], [0, 0, 10, 1]], dtype=object)
        expected = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, Fraction(1,10)]], dtype=object)
        self.assertTrue((auto_multiply(matrix) == expected).all())
    
    def test_auto_multiply_3x4_no_action(self):
        matrix = np.array([[0, 3, 3, 3], [1, 2, 2, 2], [1, 1, 10, 1]], dtype=object)
        expected = np.array([[0, 3, 3, 3], [1, 2, 2, 2], [1, 1, 10, 1]], dtype=object)
        self.assertTrue((auto_multiply(matrix) == expected).all())

class TestAutoAdd(unittest.TestCase):
    def test_auto_add_3x3_no_action(self):
        matrix = np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
        expected = np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
        self.assertTrue((auto_add(matrix) == expected).all())

    def test_auto_add_2x3(self):
        matrix = np.array([[2,  8, 4], [1,  -8, 10]])
        expected = np.array([[2, 8, 4], [0, -12, 8]])
        self.assertTrue((auto_add(matrix) == expected).all())

    def test_auto_add_3x4(self):
        matrix = np.array([[1, 2, 3, 1], [2, 4, 7, 2], [3, 7, 11, 8]], dtype=object)
        expected = np.array([[1, 2, 3, 1], [0, 1, 2, 5], [0, 0, 1, 0]], dtype=object)
        self.assertTrue((auto_add(matrix) == expected).all())

    def test_auto_add_3x5(self):
        matrix = np.array([[1, 0, -1, 0, 1], [0, 1, 2, -1, 3], [1, 1, 3, -1, 7]], dtype=object)
        expected = np.array([[1, 0, -1, 0, 1], [0, 1, 2, -1, 3], [0, 0, 2, 0, 3]], dtype=object)
        self.assertTrue((auto_add(matrix) == expected).all())

class TestReverseMatrix(unittest.TestCase):
    def test_reverse_matrix_3x3(self):
        matrix = np.array([[1, 2, 3], [0, 1, 2], [0, 0, 1]])
        expected = np.array([[0, 0, 1], [1, 0, 2], [2, 1, 3]])
        self.assertTrue((reverse_matrix(matrix) == expected).all())
    
    def test_reverse_matrix_3x4(self):
        matrix = np.array([[1, 2, 3, 1], [2, 4, 7, 2], [3, 7, 11, 8]], dtype=object)
        expected = np.array([[11, 7, 3, 8], [7, 4, 2, 2], [3, 2, 1, 1]], dtype=object)
        self.assertTrue((reverse_matrix(matrix) == expected).all())


class TestSolveLinearSystem(unittest.TestCase):
    def test_solve_linear_system_3x4(self):
        matrix = np.array([[45, -5, -40, 100], 
                           [-5, 35, -10, 0], 
                           [-40, -10, 65, 0]], dtype=object)
        expected = np.array([[1, 0, 0, 6], 
                             [0, 1, 0, 2], 
                             [0, 0, 1, 4]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_2(self):
        matrix = np.array([[1, -2, 1, 0], 
                           [0, 2, -8, 8], 
                           [5, 0, -5, 10]], dtype=object)
        expected = np.array([[1, 0, 0, 1], 
                             [0, 1, 0, 0], 
                             [0, 0, 1, -1]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_3(self):
        matrix = np.array([[1, 2, 3, 1], 
                           [2, 4, 7, 2], 
                           [3, 7, 11, 8]], dtype=object)
        expected = np.array([[1, 0, 0, -9], 
                             [0, 1, 0, 5], 
                             [0, 0, 1, 0]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_4(self):
        matrix = np.array([[10, -8, 0, 40],
                   [-8, 20, -6, 0],
                   [0, -6, 10, -20]], dtype=object)
        expected = np.array([[1, 0, 0, Fraction(28, 5)], 
                             [0, 1, 0, 2], 
                             [0, 0, 1, Fraction(-4, 5)]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())
    
    def test_solve_linear_system_3x2(self):
        matrix = np.array([[6, -5, 10], 
                           [-5, 17, -30]], dtype=object)
        expected = np.array([[1, 0, Fraction(20, 77)], 
                             [0, 1, Fraction(-130, 77)]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x5_not_solvable(self):
        matrix = np.array([[1, 0, -1, 0, 1], 
                           [0, 1, 2, -1, 3], 
                           [1, 1, 3, -1, 7]], dtype=object)
        expected = np.array([[1, 0, 0, 0, Fraction(5, 2)], 
                             [0, 1, 2, -1, 3], 
                             [0, 0, 1, 0, Fraction(3, 2)]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())