from cmath import inf
from prettytable import PrettyTable
import numpy as np
import copy
import sys
from fractions import Fraction
def display_vector(vector, header):
    table = PrettyTable()
    table.field_names = [header]
    for item in vector:
        table.add_row([item])
    print(table)

def display_matrix(matrix, headers):
    table = PrettyTable()
    table.field_names = headers
    for row in matrix:
        table.add_row([str(cell) for cell in row])
    print(table)
def scalar_multiply(vector, scalar):
    multiplied_vector = [scalar * element for element in vector]
    return multiplied_vector
def get_column(matrix, i):
    column = [row[i] for row in matrix]
    return column
def plus(vector1, vector2):
    result = []
    for i in range(len(vector1)):
        result.append(vector1[i] + vector2[i])
    return result
def minus(vector1, vector2):
    result = []
    for i in range(len(vector1)):
        result.append(vector1[i] - vector2[i])
    return result
def scal_multiply(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Векторы должны быть одинаковой длины")
    
    result = 0
    for i in range(len(vector1)):
        result += vector1[i] * vector2[i]
    
    return result
def multiply(matrix, vector):
    m = len(matrix)  # количество строк матрицы
    n = len(matrix[0])  # количество столбцов матрицы
    result = [0] * m  # инициализация результирующего вектора

    for i in range(m):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]

    return result
def gaussian_elimination(A, b):
    # Приведение матрицы A и вектора b к типу Fraction
    A = [[Fraction(elem) for elem in row] for row in A]
    b = [Fraction(elem) for elem in b]
    
    n = len(A)
    
    # Прямой ход метода Гаусса
    for i in range(n):
        max_row = i
        for j in range(i+1, n):
            if abs(A[j][i]) > abs(A[max_row][i]):
                max_row = j
        A[i], A[max_row] = A[max_row], A[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]
            
    # Обратный ход метода Гаусса
    x = [Fraction(0) for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))) / A[i][i]
    
    return x
def transpose(matrix):
    transposed_matrix = list(zip(*matrix))
    return transposed_matrix
def get_indices_of_zero(vector):
    indices = []
    for i in range(len(vector)):
        if vector[i] == 0:
            indices.append(i)
    return indices
def get_missing_indices(indices, length):
    missing_indices = []
    for i in range(length):
        if i not in indices:
            missing_indices.append(i)
    return missing_indices    
def select_columns(matrix, column_indices):
    num_rows = len(matrix)
    selected_columns = []
    
    for idx in column_indices:
        column = []
        for i in range(num_rows):
            column.append(matrix[i][idx])
        selected_columns.append(column)
    
    num_cols = len(column_indices)
    new_matrix = []
    
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(selected_columns[j][i])
        new_matrix.append(row)
    
    return new_matrix
def step(A, b, cost, lower_limit, upper_limit, x, indexes, iteration):
   
    A = [[Fraction(element) for element in row] for row in A]
    b = [Fraction(element) for element in b]
    cost = [Fraction(element) for element in cost]
    lower_limit = [Fraction(element) for element in lower_limit]
    upper_limit = [Fraction(element) for element in upper_limit]
    x = [Fraction(element) for element in x]
    indexes_b = get_missing_indices(indexes, len(cost))
    A_n = select_columns(A, indexes)
    A_b = select_columns(A, indexes_b)
    #---------- Подписи
    print("\nШАГ " + str(iteration) + '\n\nA_b:')
    display_matrix(A_b, [x + 1 for x in indexes_b])
    display_vector(x, "X_" + str(iteration))
    iteration += 1
    # -----------
    u = gaussian_elimination(transpose(A_b), [cost[idx] for idx in indexes_b])
    estimates = []
    print("------------------------------\n1)Считаем вектор потенциалов u:")
    display_vector(u, "U")
    #Считаем оценки
    box = 0
    for i in range(len(A[0])):
        if(i == indexes[box]):
            estimates.append(cost[i] - scal_multiply(get_column(A, i), u))
            if(box + 1 < len(indexes)): 
                box += 1
    print("2)Считаем оценки для небазисных компонент")
    display_vector(estimates, [x + 1 for x in indexes])
    #Находим из них максимальную по модулю, которая не удовлетворяет критери
    max_value = 0 
    is_Good = 0
    j0 = -inf
    J_minus = []
    for i in range(len(estimates)):
        if(estimates[i] < 0 and x[indexes[i]] == lower_limit[indexes[i]]):
           continue
        if(estimates[i] > 0 and x[indexes[i]] == upper_limit[indexes[i]]):
           continue
        if(estimates[i] == 0 and (x[indexes[i]] == upper_limit[indexes[i]] 
                                 or x[indexes[i]] == lower_limit[indexes[i]])):
           continue
        J_minus.append(indexes[i] + 1)
        is_Good = 1
        if(abs(max_value) <= abs(estimates[i])):
            max_value = estimates[i]
            j0 = indexes[i]
            if(max_value < 0): j0 = abs(j0)*-1
    if(is_Good == 0 or j0 == -inf):
        return x
    display_vector(J_minus, "Не выполняются")
    # j0 = input("Какой элемент взять? ")
    # j0 = int(j0) - 1
    # max_value = estimates[indexes.index(abs(j0))]
    print("3) Находим из них максимальную по модулю, которая не удовлетворяет критерию:\nj_" +
          str(abs(j0) + 1) + " со значением " + str(max_value) + "\n4)Считаем вектор допустимых направлений:\nl_b")
    # Считаем направления
    l_b = gaussian_elimination(A_b, get_column(A, abs(j0)))
    box = 0
    l = []
    for i in range(len(cost)):
        if(i == indexes_b[box]):
            l.append((-1)*l_b[box])
            if(box + 1 < len(indexes_b)): 
                box += 1
        else:
            l.append(0)
    
    l[abs(j0)] = 1
    l = scalar_multiply(l, abs(max_value)/max_value)
    l = [Fraction(element) for element in l]
    display_vector(l_b, [x + 1 for x in indexes_b])
    display_vector(l, "l")
    print("5)Считаем шаги:")
    # Считаем шаги
    q = []
    ind_of_q = []
    current_step = inf
    q0 = 0
    for i in range(len(l)):
        if(l[i] == 0): 
            continue
        if(l[i] < 0):
            q.append((lower_limit[i] - x[i])/ l[i])
            ind_of_q.append(i+1)
            if(current_step > ((lower_limit[i] - x[i])/ l[i])): 
                q0 = i
                current_step = (lower_limit[i] - x[i])/ l[i]
        if(l[i] > 0):
            q.append((upper_limit[i] - x[i])/ l[i])
            ind_of_q.append(i+1)
            if(current_step > ((upper_limit[i] - x[i])/ l[i])):
                q0 = i
                current_step = (upper_limit[i] - x[i])/ l[i]
    q = [Fraction(element) for element in q]
    display_vector(q, ind_of_q)
    print("6)Переход к следующему шагу\nШаг Q = " + str(current_step) + 
          "\n" + str(j0 + 1) + " компонента переходит в базис вместо " + str(q0+1))
    #Переход к следующему шагу
    sc = scalar_multiply(l, current_step)
    x = plus(x, sc)
    indexes[indexes.index(abs(j0))] = q0
    indexes.sort()
    return step(A, b, cost, lower_limit, upper_limit, x, indexes, iteration)
with open('/Users/iisuos/МО/output.txt', 'w') as f:
    # Сохраняем оригинальный поток вывода
    original_stdout = sys.stdout
    # Перенаправляем вывод в файл
    sys.stdout = f
    # sys.stdout = original_stdout
    # A =             [[1, 2, 2, 3],
    #                 [1, 1, 1, 4],]
    # b = [32, 40]
    # cost = [6, 5, 11, 10]
    # lower_limit = [0, 0, 0, 0]
    # upper_limit = [99999, 99999, 99999, 99999]
    # Восстанавливаем оригинальный поток вывода
    # sys.stdout = original_stdout
    # A =             [[2, 4, 4, -1, 0, 0],
    #                 [3, 2, 6, 0, -1, 0],
    #                 [6, 4, 8, 0, 0, -1]]
    # b = [4, 5, 4]
    # cost = [-240, -200, -160, 0, 0, 0]
    # lower_limit = [0, 0, 0, 0, 0, 0]
    # upper_limit = [99999, 99999, 99999, 99999, 99999, 99999]
    # A =             [[0, 2, 1, 0, 4],
    #                 [0, 0, 3, 2, 0],
    #                 [2, 0, 0, 0, 1]]
    # b = [25, 35, 9]
    # cost = [4, 8, 7, 6, 16]
    # lower_limit = [3, 5, 6, 1, 1]
    # upper_limit = [5, 8, 10, 4, 3]
    A =             [[0, 0, 0, 1, 2],
                    [3, 0, 2, 0, -4],
                    [0, -1, 3, 0, 0]]
    b = [-4, 25, 6]
    cost = [6, 3, -2, -1, -14]
    lower_limit = [1, -1, -2, 1, -3]
    upper_limit = [4, 3, 2, 4, 1]
    # A =             [[5, 1, 1, 2],
    #                 [4, 1, 2, 3]]
    # b = [50, 70]
    # cost = [-9, -4, -5, -10]
    # lower_limit = [0, 0, 0, 0]
    # upper_limit = [10000, 10000, 10000, 10000]
    # prepare to I fase
    A_f1 = copy.copy(A)
    b_f1 = copy.copy(b)
    lower_limit_f1 = copy.copy(lower_limit)
    upper_limit_f1 = copy.copy(upper_limit)
    x = copy.copy(lower_limit_f1)
    w = minus(b_f1, multiply(A_f1, x))
    for i in range(len(w)):
            # if(w[i] == 0): continue
            lower_limit_f1.append(0)
            upper_limit_f1.append(abs(w[i]))
            vector = [0] * len(b)
            if(w[i] == 0): 
                vector[i] = 1
            else:
                vector[i] = abs(w[i])/w[i]
            new_matrix = []
            for i in range(len(A_f1)):
                new_row = A_f1[i] + [vector[i]]
                new_matrix.append(new_row)
            A_f1 = new_matrix
    x += upper_limit_f1[-(len(A_f1[0]) - len(A[0])):]
    cost_f1 = [0] * len(A[0]) + [-1] * (len(A_f1[0]) - len(A[0]))
    # 1 fase
    indexes_n = get_indices_of_zero(cost_f1)
    print("I ФАЗА ")
    x = step(A_f1, b_f1, cost_f1, lower_limit_f1, upper_limit_f1, x, indexes_n, 1)
    x = x[:-(len(A_f1[0]) - len(A[0]))]
    indexes_n = indexes_n[:-(len(A_f1[0]) - len(A[0]))]
    print("II ФАЗА ")
    x = step(A, b, cost, lower_limit, upper_limit, x, indexes_n, 1)
    print("\nИтоговый план:")
    display_vector(x, "X")
    # print("Макс. значение целевой функции: " + str(scal_multiply(x, cost)))
    # sys.stdout = original_stdout
    # print(multiply(A, x))
    # x = [4, 3, -2, 1, -3]

    # display_vector(multiply(A, x), "b")
    
    # print(scal_multiply(x,cost))
