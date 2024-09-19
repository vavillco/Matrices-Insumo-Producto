import numpy as np
from scipy.linalg import solve_triangular

def calcularLU(A):
    m = A.shape[0]
    n = A.shape[1]
    c = A.copy()
    P = np.eye(n)
    
    if m != n:
        print('Matriz no cuadrada')
        return

    L = np.eye(n) 
        
    for i in range(0, n):   
        max_row = np.argmax(np.abs(Ac[i:n, i])) + i
        if i != max_row:
            # Intercambia las filas en Ac
            Ac[[i, max_row], :] = Ac[[max_row, i], :]
            
            # Intercambia las filas en P
            P[[i, max_row], :] = P[[max_row, i], :]
            
            # Intercambia las filas en L hasta la columna i (excluyendo la diagonal)
            if i > 0:
                L[[i, max_row], :i] = L[[max_row, i], :i]
            
        for j in range(i+1, n):
            piv = Ac[j][i] / Ac[i][i]
            Ac[j] = Ac[j] - piv * Ac[i]
            L[j][i] = piv

    U = Ac
    return L, U, P


def inversaLU(L, U, P):
    n = L.shape[0]
    M_identidad = []
    Inv = np.zeros((n, 0)) 

    if P is None:
        P = np.eye(n) 

    # Resolver el sistema para cada columna de la identidad permutada
    for i in range(n):
        b = P[:, i]  # Columna i de la matriz de permutación
        x = solve_triangular(L, b, lower=True)
        y = solve_triangular(U, x, lower=False)
        Inv = np.column_stack((Inv, y))  # Agregar la columna de la inversa

    return Inv
