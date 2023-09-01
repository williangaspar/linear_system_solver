import numpy as np
from fractions import Fraction


def swap_row(matrix, row_A, row_B):
    temp_row = matrix[row_A].copy()
    matrix[row_A] = matrix[row_B]
    matrix[row_B] = temp_row

    return matrix


def check_is_best_pivot(row, last_pivot_index):
    if row[last_pivot_index] != 0:
        for index in range(last_pivot_index):
            if row[index] != 0:
                return False
        return True
    else:
        return False


def find_any_pivot_index_row(row, last_pivot_index):
    if last_pivot_index == -1 and row[0] != 0:
        return 0

    if last_pivot_index == -1 and row[0] == 0:
        return -1

    if last_pivot_index == 0 and row[0] != 0:
        return -1

    if last_pivot_index > 0 and row[: last_pivot_index + 1].max() != 0:
        return -1

    rows = len(row)

    for index in range(last_pivot_index + 1, rows - 1):
        if row[index] != 0:
            return index
    return -1


def find_best_pivot_index(matrix, row_index):
    rows = len(matrix)

    for index in range(rows):
        row = matrix[index]
        if check_is_best_pivot(row, row_index):
            return index
    return -1


def find_any_pivot_index(matrix, row_index, previous_pivot_index):
    rows = len(matrix)

    for index in range(row_index, rows):
        row = matrix[index]
        if find_any_pivot_index_row(row, previous_pivot_index) > -1:
            return index
    return -1


def check_is_pivot(row, index):
    if row[index] != 0:
        for index in range(index):
            if row[index] != 0:
                return False
        return True
    else:
        return False


def find_pivot_index(matrix, row_index):
    rows = len(matrix)

    for index in range(rows):
        row = matrix[index]
        if check_is_pivot(row, row_index):
            return index
    return -1


def push_zero_row_to_the_end(matrix):
    rows, columns = matrix.shape

    floating_index = -1
    real_index = -1
    stop_index = rows * 2

    while real_index != stop_index and floating_index != rows - 1:
        floating_index += 1
        real_index += 1
        row = matrix[floating_index]

        if row[: columns - 1].max() == 0:
            matrix = np.delete(matrix, floating_index, 0)
            matrix = np.append(matrix, [row], axis=0)
            floating_index = -1

    return matrix


def auto_swap(matrix):
    rows = len(matrix)

    # Swaps best case scenario
    for row_index in range(rows):
        row = matrix[row_index]
        if check_is_pivot(row, row_index) == False:
            pivot_index = find_pivot_index(matrix, row_index)
            if pivot_index > -1:
                matrix = swap_row(matrix, row_index, pivot_index)

    # Swaps any case scenario
    for row_index in range(rows):
        row = matrix[row_index]
        if check_is_pivot(row, row_index) == False:
            pivot_index = find_any_pivot_index(matrix, row_index, row_index)
            if pivot_index > -1:
                matrix = swap_row(matrix, row_index, pivot_index)
    return matrix


def auto_multiply(matrix):
    rows = len(matrix)

    for row_index in range(rows):
        current_row = matrix[row_index]
        cell_value = current_row[row_index]

        is_pivot = check_is_pivot(current_row, row_index)
        if is_pivot and cell_value != 1:
            matrix[row_index] = Fraction(1, cell_value) * current_row
    return matrix


def auto_add(matrix):
    rows = len(matrix)

    pivot_index = -1

    for row_index in range(rows - 1):
        current_row = matrix[row_index]
        pivot_index = find_any_pivot_index(matrix, row_index, pivot_index)
        if pivot_index == -1:
            continue
        for column_index in range(row_index + 1, rows):
            cell_value = matrix[column_index][pivot_index]
            current_row_value = current_row[pivot_index]

            if cell_value != 0 and current_row_value != 0:
                row_multiplier = Fraction(-cell_value, current_row_value)
                matrix[column_index] = (
                    row_multiplier * current_row + matrix[column_index]
                )
    return matrix


def reverse_matrix(matrix):
    rows, columns = matrix.shape
    matrix = np.flipud(matrix)

    for row_index in range(rows):
        row = matrix[row_index]
        matrix[row_index] = np.append(np.flip(row[: columns - 1]), row[columns - 1])
    return matrix


def solve_top_to_bottom(matrix):
    matrix = push_zero_row_to_the_end(matrix)
    matrix = auto_swap(matrix)
    matrix = auto_multiply(matrix)
    matrix = auto_add(matrix)

    matrix = auto_swap(matrix)
    matrix = push_zero_row_to_the_end(matrix)
    matrix = auto_multiply(matrix)

    return matrix


def solve_bottom_to_top(matrix):
    matrix = reverse_matrix(matrix)
    matrix = auto_add(matrix)
    matrix = auto_multiply(matrix)

    matrix = reverse_matrix(matrix)
    matrix = auto_swap(matrix)
    matrix = push_zero_row_to_the_end(matrix)
    matrix = auto_add(matrix)
    matrix = auto_multiply(matrix)
    matrix = push_zero_row_to_the_end(matrix)

    return matrix


def solve_linear_system(matrix):
    matrix = solve_top_to_bottom(matrix)
    matrix = solve_bottom_to_top(matrix)

    return matrix
