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
import scipy as sp
from scipy.spatial import ConvexHull
def data_extended(point_xy, point_z, extend_distance = 2., extended_mode = "normal to boundary", boundary = "auto"):
    if boundary == "auto":
        center = np.mean(point_xy, axis = 0)
        hull = ConvexHull(point_xy)
        # hull.vertices
        # hull.equations

# def get_dyadic_cubic_partion_tree(*arg, dim = 3):
#     assert dim == 2, dim == 3
#     if dim == 3:
#         L = len(arg)
#         assert L > 0 and L <= 3
#         if L == 1:
#             points = arg[0].copy()
#         if else:
#             points = np.column_stack(list(arg)).copy()
#         assert points.shape[1] == 3, "wrong shape input"

#         # class tree():
#         #     def __init__(self, parent = None):
#         #         super(node, self).__init__()
#         #         self.arg = arg
#         #         self.children = [None for i in range(8)]
#         #         self.parent = parent
#         1

def nbd_detecter(pts, detect_pts, detect_R, dim = 2, maximal_possible_detect_R = 0.3):
    if dim == 2:
        import time
        start_time = time.time()

        # pts = (np.random.rand(10**6 * 2, 2) * 2 - 1)
        # detect_pts = (np.random.rand((2 * 100)**2, 2) * 2 - 1) * 1.2
        pp = np.linspace(0, 100.1, len(detect_pts)).astype(int)
        filter_pp = np.diff(np.append(pp, 100)) > 0
        filter_pp[-1] = True
        # maximal_possible_detect_R = 0.3
        # detect_R = 0.2

        partition_pitch = 0.1
        partition_x = []
        detect_min_x = np.min(detect_pts[:, 0])
        detect_total_distance_x = (np.max(detect_pts[:, 0]) - detect_min_x)
        min_x = np.min(pts[:, 0])
        total_distance_x = (np.max(pts[:, 0]) - min_x)
        total_partition_x = int(total_distance_x / partition_pitch) if total_distance_x > partition_pitch else 1
        partition_pitch_x = total_distance_x / total_partition_x
        temp = np.arange(len(pts))
        for i in range(total_partition_x):
            TF = pts[temp, 0] <= partition_pitch_x * (i + 1) + min_x
            partition_x.append(temp[TF])
            temp = temp[~TF]
        number_of_min_side_extended_x = int(max(min_x - detect_min_x, 0) // partition_pitch_x + \
                                            maximal_possible_detect_R // partition_pitch_x + 10)
        number_of_max_side_extended_x = int(max((detect_min_x + detect_total_distance_x) - \
                                            (min_x + total_distance_x), 0) // partition_pitch_x + \
                                            maximal_possible_detect_R // partition_pitch_x + 10)
        # number_of_min_side_extended_x = 0
        # number_of_max_side_extended_x = 0
        partition_x = [np.empty(0)] * number_of_min_side_extended_x + \
                                            partition_x + \
                                            [np.empty(0)] * number_of_max_side_extended_x


        # partition_y = []
        detect_min_y = np.min(detect_pts[:, 0])
        detect_total_distance_y = (np.max(detect_pts[:, 0]) - detect_min_y)
        min_y = np.min(pts[:, 1])
        total_distance_y = (np.max(pts[:, 1]) - min_y)
        total_partition_y = int(total_distance_y / partition_pitch) if total_distance_y > partition_pitch else 1
        partition_pitch_y = total_distance_y / total_partition_y
        number_of_min_side_extended_y = int(max(min_y - detect_min_y, 0) // partition_pitch_y + \
                                            maximal_possible_detect_R // partition_pitch_y + 10)
        number_of_max_side_extended_y = int(max((detect_min_y + detect_total_distance_y) - \
                                            (min_y + total_distance_y), 0) // partition_pitch_y + \
                                            maximal_possible_detect_R // partition_pitch_y + 10)
        # number_of_min_side_extended_y = 0
        number_of_max_side_extended_y = 0
        partition_xy = np.array([[0] * (total_partition_y + \
                                            number_of_min_side_extended_y + \
                                            number_of_max_side_extended_y)
                                ] * (total_partition_x + \
                                            number_of_min_side_extended_x + \
                                            number_of_max_side_extended_x), dtype = object)
        for i in range(number_of_min_side_extended_x, number_of_min_side_extended_x + total_partition_x):
            temp = partition_x[i].copy()
            for j in range(number_of_min_side_extended_y, number_of_min_side_extended_y + total_partition_y):
                TF = pts[temp, 1] <= partition_pitch_y * (j + 1 - number_of_min_side_extended_y) + min_y
                # partition_y.append(temp[TF])
                partition_xy[i, j] = temp[TF].copy()
                # print(j, pts[temp, 1])
                temp = temp[~TF]
                # print(temp)
                # print(i, j, partition_xy[i, j])
                # if j > 2:
                    # break
            # break
        for i in range(number_of_min_side_extended_x):
            for j in range(number_of_min_side_extended_y + total_partition_y + number_of_max_side_extended_y):
                partition_xy[i, j] = np.empty(0)
        for i in range(number_of_min_side_extended_x + total_partition_x, \
                    number_of_min_side_extended_x + total_partition_x + number_of_max_side_extended_x):
            for j in range(number_of_min_side_extended_y + total_partition_y + number_of_max_side_extended_y):
                partition_xy[i, j] = np.empty(0)
        for i in range(number_of_min_side_extended_x, number_of_min_side_extended_x + total_partition_x):
            for j in range(number_of_min_side_extended_y):
                partition_xy[i, j] = np.empty(0)
        for i in range(number_of_min_side_extended_x, number_of_min_side_extended_x + total_partition_x):
            for j in range(number_of_min_side_extended_y + total_partition_y, \
                            number_of_min_side_extended_y + total_partition_y + number_of_max_side_extended_y):
                partition_xy[i, j] = np.empty(0)

        temp = (int(2 * detect_R / partition_pitch) + 1)
        nbd_partition_id = (temp + 1) // 2 if temp % 2 else temp // 2
        nbd_of_detect_pts_ids = []
        print("\nprepare time:", time.time() - start_time)
        for i in range(len(detect_pts)):
            if filter_pp[i]:
                print(pp[i], end = "\r")
            partition_id_x = int((detect_pts[i, 0] - min_x) / partition_pitch_x) + number_of_min_side_extended_x
            partition_id_y = int((detect_pts[i, 1] - min_y) / partition_pitch_y) + number_of_min_side_extended_y
            # print("\t detect_pts:", detect_pts[i])
            # print(pts[partition_x[partition_id_x]])

            temp_xy = partition_xy[max(partition_id_x - nbd_partition_id, 0):min(partition_id_x + nbd_partition_id + 1, \
                                                           number_of_min_side_extended_x + \
                                                           number_of_max_side_extended_x + \
                                                           total_partition_x), 
                                    max(partition_id_y - nbd_partition_id, 0):min(partition_id_y + nbd_partition_id + 1, \
                                                           number_of_min_side_extended_y + \
                                                           number_of_max_side_extended_y + \
                                                           total_partition_y)]

            # s_time = time.time()
            temp_xy = np.concatenate(list(temp_xy.reshape(-1))).astype(int)
            nbd_of_detect_pts_ids.append(temp_xy)
            if len(nbd_of_detect_pts_ids[-1]):
                temp_ = pts[nbd_of_detect_pts_ids[-1], :] - detect_pts[i]
                nbd_of_detect_pts_ids[-1] = nbd_of_detect_pts_ids[-1][np.linalg.norm(temp_, axis = 1) < detect_R]

            # temp = pts - detect_pts[i]
            # temp = pts[np.linalg.norm(temp, axis = 1) < detect_R, :]
            # print(time.time() - s_time)

        print("\nALL:", time.time() - start_time)

