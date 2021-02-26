import numpy as np
import os
import gc

def complex2real(arg, output_stack = True):
    arg = arg.copy()
    A = [np.real(arg), np.imag(arg)]
    if not output_stack:
        return A[0], A[1]
    return np.column_stack([A[0], A[1]])

def real2complex(*arg):
    if len(arg) == 1:
        assert type(arg[0]) == np.ndarray
        arg = arg[0]
        S = arg.shape
        assert S[-1] == 2
        A = [arg[..., 0].copy(), arg[..., 1].copy()]
    else:
        assert len(arg) == 2
        A = [arg[0].copy(), arg[1].copy()]
    return A[0] + A[1] * 1J

def car2pol(*arg, output_column_stack = True, output_radius_first = True, degree = "deg"):
    if len(arg) == 1:
        assert type(arg[0]) == np.ndarray
        arg = arg[0]
        S = arg.shape
        assert len(S) == 2
        assert S[1] == 2
        C = real2complex(arg[0])
    else:
        assert len(arg) == 2
        C = real2complex(arg[0], arg[1])
    P = [np.abs(C), np.angle(C, deg = (degree == "deg"))]
    if not output_radius_first:
        P[0], P[1] = P[1], P[0]
    if not output_column_stack:
        return P[0], P[1]
    return np.column_stack([P[0], P[1]])

def pol2car(*arg, output_column_stack = True, input_radius_first = True, degree = "deg"): # degree = deg or rad
    if len(arg) == 1:
        assert type(arg[0]) == np.ndarray
        arg = arg[0]
        S = arg.shape
        assert len(S) == 2
        assert S[1] == 2
        arg = arg.copy()
        arg = [arg[:, 0], arg[:, 1]]
    else:
        assert len(arg) == 2
        arg = [arg[0], arg[1]]
    if not input_radius_first:
        arg[0], arg[1] = arg[1], arg[0]
    Rs = arg[0].copy()
    Thetas = arg[1].copy()
    if degree == "deg":
        Thetas = np.deg2rad(Thetas)
    return complex2real(Rs * np.exp(Thetas * 1J), output_stack = output_column_stack)

# from reader import UA3P_reader
# UA3P_xy, UA3P_z, UA3P_zd, _, _ = UA3P_reader("XX.csv", dim = 3)
# def data_extended(point_xy, point_z, boundary = "auto"):

def get_dyadic_cubic_partion_tree(*arg, dim = 3):
    assert dim == 2, dim == 3
    if dim == 3:
        L = len(arg)
        assert L > 0 and L <= 3
        if L == 1:
            points = arg[0].copy()
        if else:
            points = np.column_stack(list(arg)).copy()
        assert points.shape[1] == 3, "wrong shape input"

        1

