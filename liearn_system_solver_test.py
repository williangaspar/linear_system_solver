import unittest
import numpy as np
from fractions import Fraction

from linear_sistem_solver import (
    auto_add,
    auto_multiply,
    auto_swap,
    check_is_pivot,
    find_any_pivot_index,
    find_any_pivot_index_row,
    push_zero_row_to_the_end,
    solve_linear_system,
    swap_row,
)


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


class TestFindAnyPivotIndexRow(unittest.TestCase):
    def test_find_any_pivot_index_2x2(self):
        row = np.array([1, 2])
        self.assertEqual(find_any_pivot_index_row(row, -1), 0)

    def test_find_any_pivot_index_2x2_2(self):
        row = np.array([0, 2])
        self.assertEqual(find_any_pivot_index_row(row, 0), -1)

    def test_find_any_pivot_index_row_2x2_3(self):
        row = np.array([1, 2])
        # we already have a pivot at index 0
        self.assertEqual(find_any_pivot_index_row(row, 0), -1)

    def test_find_any_pivot_index_row_3x3(self):
        row = np.array([0, 3, 0])
        self.assertEqual(find_any_pivot_index_row(row, 0), 1)

    def test_find_any_pivot_index_row_3x3_2(self):
        row = np.array([0, 0, 3])
        self.assertEqual(find_any_pivot_index_row(row, 1), -1)

    def test_find_any_pivot_index_row_3x3_3(self):
        row = np.array([0, 0, 3])
        self.assertEqual(find_any_pivot_index_row(row, 2), -1)

    def test_find_any_pivot_index_row_4x3_3(self):
        row = np.array([0, 0, 3, 9])
        self.assertEqual(find_any_pivot_index_row(row, 2), -1)

    def test_find_any_pivot_index_row_4x3_4(self):
        row = np.array([0, 0, 3, 9])
        self.assertEqual(find_any_pivot_index_row(row, 1), 2)

    def test_find_any_pivot_index_row_4x3_5(self):
        row = np.array([0, 0, 9, 0])
        self.assertEqual(find_any_pivot_index_row(row, 1), 2)

    def test_find_any_pivot_index_row_4x3_6(self):
        row = np.array([0, 0, 0, 0])
        self.assertEqual(find_any_pivot_index_row(row, 2), -1)

    def test_find_any_pivot_index_row_4x3_7(self):
        row = np.array([0, 0, 0, 0])
        self.assertEqual(find_any_pivot_index_row(row, -1), -1)


class TestFindAnyPivotIndex(unittest.TestCase):
    def test_find_any_pivot_index_2x2(self):
        matrix = np.array([[1, 2], [3, 4]])
        self.assertEqual(find_any_pivot_index(matrix, 0, -1), 0)

    def test_find_any_pivot_index_2x2_2(self):
        matrix = np.array([[0, 2], [1, 0]])
        self.assertEqual(find_any_pivot_index(matrix, 0, -1), 1)

    def test_find_any_pivot_index_2x2_3(self):
        matrix = np.array([[0, 0], [0, 0]])
        self.assertEqual(find_any_pivot_index(matrix, 0, -1), -1)

    def test_find_any_pivot_index_3x3(self):
        matrix = np.array([[0, 0, 3], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(find_any_pivot_index(matrix, 0, -1), 1)

    def test_find_any_pivot_index_3x3_2(self):
        matrix = np.array([[0, 0, 3], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(find_any_pivot_index(matrix, 1, -1), 1)


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


class TestPushZeroRowToTheEnd(unittest.TestCase):
    def test_push_zero_row_to_the_end_2x2(self):
        matrix = np.array([[0, 0], [1, 2]])
        expected = np.array([[1, 2], [0, 0]])
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_3x3(self):
        matrix = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]])
        expected = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_4x4(self):
        matrix = np.array([[0, 0, 0, 0], [1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4]])
        expected = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [0, 0, 0, 0]]
        )
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_4x4_2(self):
        matrix = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [-1, -2, -3, -4], [5, 6, 7, 8]])
        expected = np.array(
            [[-1, -2, -3, -4], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_4x4_3(self):
        matrix = np.array([[0, 0, 0, 1], [-1, -2, -3, -4], [5, 6, 7, 8], [0, 0, 0, 0]])
        expected = np.array(
            [[-1, -2, -3, -4], [5, 6, 7, 8], [0, 0, 0, 1], [0, 0, 0, 0]]
        )
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_4x4_4(self):
        matrix = np.array([[1, 2, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0], [5, 6, 7, 8]])
        expected = np.array([[1, 2, 2, 1], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_5x4_4(self):
        matrix = np.array(
            [[1, 2, 2, 1], [5, 6, 7, 8], [0, 0, 0, 0], [0, 0, 0, 0], [1, 8, 2, 1]]
        )
        expected = np.array(
            [[1, 2, 2, 1], [5, 6, 7, 8], [1, 8, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())

    def test_push_zero_row_to_the_end_5x4_4_2(self):
        matrix = np.array(
            [[1, 2, 2, 1], [0, 6, 7, 8], [0, 0, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        expected = np.array(
            [[1, 2, 2, 1], [0, 6, 7, 8], [0, 0, 2, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.assertTrue((push_zero_row_to_the_end(matrix) == expected).all())


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
        matrix = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [0, 0, 3, 0]])
        expected = np.array([[1, 2, 3, 4], [0, 0, 3, 0], [0, 0, 0, 0]])
        self.assertTrue((auto_swap(matrix) == expected).all())


class TestAutoMultiply(unittest.TestCase):
    def test_auto_multiply_3x3(self):
        matrix = np.array([[3, 3, 3], [0, 2, 2], [1, 1, 1]])
        expected = np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]])
        self.assertTrue((auto_multiply(matrix) == expected).all())

    def test_auto_multiply_3x4(self):
        matrix = np.array([[3, 3, 3, 3], [0, 2, 2, 2], [0, 0, 10, 1]], dtype=object)
        expected = np.array(
            [[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, Fraction(1, 10)]], dtype=object
        )
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
        matrix = np.array([[2, 8, 4], [1, -8, 10]])
        expected = np.array([[2, 8, 4], [0, -12, 8]])
        self.assertTrue((auto_add(matrix) == expected).all())

    def test_auto_add_3x5(self):
        matrix = np.array(
            [[1, 0, -1, 0, 1], [0, 1, 2, -1, 3], [1, 1, 3, -1, 7]], dtype=object
        )
        expected = np.array(
            [[1, 0, -1, 0, 1], [0, 1, 2, -1, 3], [0, 0, 2, 0, 3]], dtype=object
        )
        self.assertTrue((auto_add(matrix) == expected).all())


class TestSolveLinearSystem(unittest.TestCase):
    def test_solve_linear_system_3x4(self):
        matrix = np.array(
            [[45, -5, -40, 100], [-5, 35, -10, 0], [-40, -10, 65, 0]], dtype=object
        )
        expected = np.array([[1, 0, 0, 6], [0, 1, 0, 2], [0, 0, 1, 4]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_2(self):
        matrix = np.array([[1, -2, 1, 0], [0, 2, -8, 8], [5, 0, -5, 10]], dtype=object)
        expected = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, -1]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_3(self):
        matrix = np.array([[1, 2, 3, 1], [2, 4, 7, 2], [3, 7, 11, 8]], dtype=object)
        expected = np.array([[1, 0, 0, -9], [0, 1, 0, 5], [0, 0, 1, 0]], dtype=object)
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x4_4(self):
        matrix = np.array(
            [[10, -8, 0, 40], [-8, 20, -6, 0], [0, -6, 10, -20]], dtype=object
        )
        expected = np.array(
            [[1, 0, 0, Fraction(28, 5)], [0, 1, 0, 2], [0, 0, 1, Fraction(-4, 5)]],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x2(self):
        matrix = np.array([[6, -5, 10], [-5, 17, -30]], dtype=object)
        expected = np.array(
            [[1, 0, Fraction(20, 77)], [0, 1, Fraction(-130, 77)]], dtype=object
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_3x5_free_variable(self):
        matrix = np.array(
            [[1, 0, -1, 0, 1], [0, 1, 2, -1, 3], [1, 1, 3, -1, 7]], dtype=object
        )
        expected = np.array(
            [
                [1, 0, 0, 0, Fraction(5, 2)],
                [0, 1, 0, -1, 0],
                [0, 0, 1, 0, Fraction(3, 2)],
            ],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_5x4_empty_lines(self):
        matrix = np.array(
            [
                [1, 2, 0, 4],
                [2, 1, -7, 1],
                [-1, 3, 6, 2],
                [3, 1, -11, 1],
                [2, 2, -2, 6],
            ],
            dtype=object,
        )
        expected = np.array(
            [
                [1, 0, 0, 4],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_6_4(self):
        matrix = np.array(
            [
                [1, 2, 7, 1, -1, -15],
                [1, 1, 3, 1, 0, -6],
                [3, 2, 5, -1, 9, 19],
                [1, -1, -5, 2, 0, 5],
            ],
            dtype=object,
        )
        expected = np.array(
            [
                [1, 0, -1, 0, 3, 10],
                [0, 1, 4, 0, -1, -9],
                [0, 0, 0, 1, -2, -7],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_6x5(self):
        matrix = np.array(
            [
                [2, 1, 5, 1, 5, 1],
                [1, 1, 3, 1, 6, -1],
                [-1, 1, -1, 0, 4, -3],
                [-3, 2, -4, -4, -7, 0],
                [3, -1, 5, 2, 2, 3],
            ],
            dtype=object,
        )
        expected = np.array(
            [
                [1, 0, 2, 0, -1, 2],
                [0, 1, 1, 0, 3, -1],
                [0, 0, 0, 1, 4, -2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())

    def test_solve_linear_system_5x4_empty_line(self):
        matrix = np.array(
            [
                [2, -1, 3, 9],
                [1, 1, 1, 2],
                [3, 0, -1, 1],
                [2, 1, -2, -3],
            ],
            dtype=object,
        )
        expected = np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, -1],
                [0, 0, 1, 2],
                [0, 0, 0, 0],
            ],
            dtype=object,
        )
        self.assertTrue((solve_linear_system(matrix) == expected).all())
