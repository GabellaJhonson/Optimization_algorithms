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
def step(A, b, cost, lower_limit, upper_limit, indexes_n, iteration):
   
    A = [[Fraction(element) for element in row] for row in A]
    b = [Fraction(element) for element in b]
    cost = [Fraction(element) for element in cost]
    lower_limit = [Fraction(element) for element in lower_limit]
    upper_limit = [Fraction(element) for element in upper_limit]
    indexes_b = get_missing_indices(indexes_n, len(cost))
    A_n = select_columns(A, indexes_n)
    A_b = select_columns(A, indexes_b)
    #---------- Подписи
    print("\nШАГ " + str(iteration) + '\n\nA_b:')
    display_matrix(A_b, [x + 1 for x in indexes_b])
    iteration += 1
    # -----------
    u = gaussian_elimination(transpose(A_b), [cost[idx] for idx in indexes_b])
    estimates = []
    print("------------------------------\n1)Считаем вектор потенциалов u:")
    display_vector(u, "U")
    #Считаем оценки
    box = 0
    for i in range(len(A[0])):
        if(i == indexes_n[box]):
            estimates.append(cost[i] - scal_multiply(get_column(A, i), u))
            if(box + 1 < len(indexes_n)): 
                box += 1
    print("2)Считаем оценки для небазисных компонент")
    display_vector(estimates, [x + 1 for x in indexes_n])
    #Формируем множества J_+ и J_-
    J_plus = []
    J_minus = []
    for i in range(len(estimates)):
        if(estimates[i] < 0):
           J_minus.append(indexes_n[i])
        if(estimates[i] >= 0):
           J_plus.append(indexes_n[i])
    # Строим псевдоплан ae
    ae_n = []
    for i in range(len(estimates)):
        if(estimates[i] < 0):
           ae_n.append(lower_limit[indexes_n[i]])
        if(estimates[i] >= 0):
           ae_n.append(upper_limit[indexes_n[i]])
    ae_b = gaussian_elimination(A_b, minus(b, multiply(A_n, ae_n)))
    #Проверим условие оптимальности]
    p = []
    for i in range(len(indexes_b)):
        if(ae_b[i] <= upper_limit[indexes_b[i]] and ae_b[i] >= lower_limit[indexes_b[i]]):
            continue
        p.append(indexes_b[i])
    ae = ae_n + ae_b
    indexes_nb = indexes_n + indexes_b
    sort_vec = [0]* len(indexes_nb)
    for i in range(len(indexes_nb)):
        sort_vec[indexes_nb[i]] = ae[i]
    display_vector(sort_vec, ["ae"])
    # if(len(p) == 0 or u == [0]*len(u)): 
    if(len(p) == 0): 
        return sort_vec
    print("3)Не выполняются кр оптимальности")
    display_vector([x + 1 for x in p], "P")
    #Найдем i0 из невыполняющих критерий
    # i0 = input("Какой элемент взять? ")
    # i0 = int(i0) - 1
    i0 = np.random.choice(p)      # !!!!!!!!!!
    #Считаем вектор l
    temp = [0] * len(indexes_b)
    if(ae_b[indexes_b.index(i0)] < lower_limit[i0]):
        temp[indexes_b.index(i0)] = 1
    else:
        temp[indexes_b.index(i0)] = -1
    l_y = gaussian_elimination(transpose(A_b), temp)
    l_n = multiply(transpose(A_n), l_y)
    for i in range(len(indexes_n)):
        if(indexes_n[i] in J_plus):
            l_n[i] *= -1
    print("4)Вектор l_n:")
    display_vector(l_n, [x + 1 for x in indexes_n])
    print("5)Считаем шаги:")
    # Считаем шаги
    q = []
    for i in range(len(l_n)):
        if(l_n[i] < 0):
            q.append(abs(estimates[i]/l_n[i]))
        if(l_n[i] >= 0):
            q.append(9999999)         
    q = [Fraction(element) for element in q]
    current_step = min(q)
    q0 = q.index(current_step)
    display_vector(q, "Q")
    #Переход к следующему шагу
    print("6)Переход к следующему шагу\nШаг Q = " + str(current_step) + 
          "\n" + str(indexes_n[q0] + 1) + " компонента переходит в базис вместо " + str(i0+1))
    indexes_n[q0] = i0
    indexes_n.sort()
    return step(A, b, cost, lower_limit, upper_limit, indexes_n, iteration)
with open('/Users/iisuos/Двойсвтенный метод/output.txt', 'w') as f:
    # Сохраняем оригинальный поток вывода
    original_stdout = sys.stdout
    # Перенаправляем вывод в файл
    sys.stdout = f
    # A =             [[1, 2, 2, 3],
    #                 [1, 1, 1, 4],]
    # b = [32, 40]
    # cost = [6, 5, 11, 10]
    # lower_limit = [0, 0, 0, 0]
    # upper_limit = [99999, 99999, 99999, 99999]
    # Восстанавливаем оригинальный поток вывода
    # sys.stdout = original_stdout
    # A =             [[2, 1, 0, 0, 0],
    #                 [0, 0, 3, 4, 0],
    #                 [3, 1, 0, 0, 2]]
    # b = [6, 11, 19]
    # cost = [11, 2, 2, 8, 6]
    # lower_limit = [-2, -1, 1, 1, 2]
    # upper_limit = [3, 4, 6, 6, 7]
    # Йода
    # A =             [[-4, 1, 0, 0, 3],
    #                 [2, 0, 0, 0, -1],
    #                 [0, 0, 4, 2, 0]]
    # b = [6, -2, 12]
    # cost = [-4, 0, 4, 5, -4]
    # lower_limit = [0, -1, 0, -2, 0]
    # upper_limit = [4, 4, 5, 3, 6]
    # Жека кри
    # A =             [[3, 0, -1, 0, 1],
    #                 [-2, 0, 0, 3, 0],
    #                 [0, 4, 0, 0, 1]]
    # b = [30, -11, 2]
    # cost = [5, 0,-4, 15,5]
    # lower_limit = [2, 0, 1, 0,-1]
    # upper_limit = [10, 4, 5, 4, 3]
    A =             [[4, 0, 0, -1, 2],
                    [0, 0, 1, 0, 3],
                    [0, 2, 0, 1, 0]]
    b = [1, 3, 3]
    cost = [12, 2, 0, 5, 3]
    lower_limit = [-1, 0, -4, 1, 1]
    upper_limit = [4, 4, 1, 5, 6]
    # A =             [[0, 3, 1, 0, 0],
    #                 [-4, 1, 0, 0, 0],
    #                 [2, 0, 0, 4, -1]]
    # b = [-4, 8, -1]
    # cost = [-10, 5, 1, 1, 1]
    # lower_limit = [-2, -3, -5, 0, 0]
    # upper_limit = [3, 2, 0, 4, 5]
    # A =             [[8, 25],
    #                 [8, 5], 
    #                 [1, 5]]
    # b = [800, 640, 145]
    # cost = [80, 70]
    # lower_limit = [3, 0]
    # upper_limit = [78, 28]
    #prepare to I fase
    A_f1 = copy.copy(A)
    b_f1 = copy.copy(b)
    lower_limit_f1 = copy.copy(lower_limit)
    upper_limit_f1 = copy.copy(upper_limit)
    x = copy.copy(lower_limit_f1)
    w = minus(b_f1, multiply(A_f1, x))
    for i in range(len(w)):
            if(w[i] == 0): continue
            lower_limit_f1.append(0)
            upper_limit_f1.append(0)
            # upper_limit_f1.append(abs(w[i]))
            vector = [0] * len(b)
            vector[i] = abs(w[i])/w[i]
            new_matrix = []

            for i in range(len(A_f1)):
                new_row = A_f1[i] + [vector[i]]
                new_matrix.append(new_row)
            A_f1 = new_matrix
    x += upper_limit_f1[-(len(A_f1[0]) - len(A[0])):]
    cost_f1 = [0] * len(A[0]) + [-1] * (len(A_f1[0]) - len(A[0]))
    # 1 fase
    indexes_n = [0, 1, 2, 3, 4]
    print("I ФАЗА ")
    # x = step(A, b, cost, lower_limit, upper_limit, [1, 2], 1)
    x = step(A_f1, b_f1, cost + [0]*3, lower_limit_f1, upper_limit_f1, indexes_n, 1)
    # x = step(A_f1, b_f1, cost_f1, lower_limit_f1, upper_limit_f1, indexes_n, 1)
    # x = step(A, b, cost, lower_limit, upper_limit, [2,4], 1)
    indexes_b = get_missing_indices(indexes_n, len(cost_f1))
    print("\nРешив задачу первой фазы, нашли начальный базис J_b:")
    print([x + 1 for x in indexes_b])
    print("\nII ФАЗА")
    x = step(A, b, cost, lower_limit, upper_limit, indexes_n[:-3], 1)
    print("\nИтоговое решение:")
    display_vector(x, "X")
    print("Макс. значение целевой функции: " + str(scal_multiply(x, cost)))
    sys.stdout = original_stdout
    # print(multiply(A, x))
    # x = {5, 0, 25, 0}
    # print(scal_multiply(x,cost))
# 3 0 1 2 5