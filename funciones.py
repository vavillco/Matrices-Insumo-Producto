import numpy as np
from scipy.linalg import solve_triangular

def calcularLU(A):
    L, U = [],[]
    P = None
    # su código
    
    ###########
    return L, U, P


def inversaLU(L, U, P=None):
    Inv = []
    # su código
    n = L.shape[0]
    M_identidad = []
    Inv = np.zeros((n, 0)) 
    for i in range(n):
        canonico = np.zeros(n)
        canonico[i] = 1
        M_identidad.append(canonico)
    for i in range(n):
        x = solve_triangular(L, M_identidad[i], lower=True)
        y = solve_triangular(U, x, lower=False)
        Inv = np.column_stack((Inv, y))  
        # capaz que no se pueda usar esta funcion, preguntar
    ###########
    return Inv
