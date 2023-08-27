from linear_sistem_solver import solve_linear_system
from matrix_printer import print_augumented_matrix
from matrix_reader import read_matrix


def main():
    matrix = read_matrix()
    if (len(matrix) == 0):
        return
    
    print_augumented_matrix(matrix)
    print()
    solved_matrix = solve_linear_system(matrix)
    print_augumented_matrix(solved_matrix)
    return

if __name__ == '__main__':
    main()