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
    if type(arg) != np.ndarray and len(arg) == 2:
        A = [arg[0].copy(), arg[1].copy()]
    else:
        if type(arg) != np.ndarray:
            assert len(arg) == 1
            arg = arg[0]
        S = arg.shape
        assert S[-1] == 2
        A = [arg[..., 0].copy(), arg[..., 1].copy()]
    return A[0] + A[1] * 1J

def car2pol(*arg, output_column_stack = True, output_radius_first = True, degree = "deg"):
    if type(arg) != np.ndarray and len(arg) == 2:
        C = real2complex(arg[0], arg[1])
    else:
        if type(arg) != np.ndarray:
            assert len(arg) == 1
            arg = arg[0]
        S = arg.shape
        assert len(S) == 2
        assert S[1] == 2
        C = real2complex(arg)
    P = [np.abs(C), np.angle(C, deg = (degree == "deg"))]
    if degree == "deg":
        P[1] = P[1] % 360
    else:
        P[1] = P[1] % (np.pi * 2)
    if not output_radius_first:
        P[0], P[1] = P[1], P[0]
    if not output_column_stack:
        return P[0], P[1]
    return np.column_stack([P[0], P[1]])

def pol2car(*arg, output_column_stack = True, input_radius_first = True, degree = "deg"): # degree = deg or rad
    if type(arg) != np.ndarray and len(arg) == 2:
        arg = [arg[0], arg[1]]
    else:
        if type(arg) != np.ndarray:
            assert len(arg) == 1
            arg = arg[0]
        S = arg.shape
        assert len(S) == 2
        assert S[1] == 2
        arg = arg.copy()
        arg = [arg[:, 0], arg[:, 1]]
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
def data_extended(point_xy, point_z, detect_R, 
    boundary = "auto", extended_mode = "normal to boundary", 
    extend_distance = 2., extend_density = 1.):
    if boundary == "auto" and extended_mode == "normal to boundary":
        center = np.mean(point_xy, axis = 0)
        degree_pitch = 1 / extend_density
        degree_pitch = degree_pitch if min(degree_pitch - (360 % degree_pitch), \
                                                (360 % degree_pitch)) < 0.01 \
                                        else 360. / (int(360. / degree_pitch) + 1.)
        unit_vectors_rad = np.deg2rad(np.arange(0., 360., degree_pitch))
        unit_vectors = complex2real(np.exp(1J * unit_vectors_rad))
        hull = ConvexHull(point_xy) # hull.equations hull.vertices
        ray_equations = unit_vectors[:, [1, 0]].copy()
        ray_equations[:, 1] = ray_equations[:, 1] * -1.
        ray_equations = np.column_stack([ray_equations, np.dot(ray_equations, center[..., None]).reshape(-1)])

        hull_vertices = point_xy[hull.vertices, :].copy()
        _, hull_vertices_rad = car2pol(hull_vertices - center, output_column_stack = False, degree = "rad")

        ray_points_on_boundary = np.zeros_like(unit_vectors)
        k = np.argmin(hull_vertices_rad) - len(hull_vertices_rad)
        k0 = k
        start_rad = hull_vertices_rad[k]
        end_rad = hull_vertices_rad[k + 1]
        temp_eq = hull_vertices[k + 1] - hull_vertices[k]
        temp_eq[0], temp_eq[1] = temp_eq[1], -temp_eq[0]
        temp_eq = temp_eq / np.linalg.norm(temp_eq)
        temp_eq = np.append(temp_eq, np.dot(temp_eq, hull_vertices[k]))
        unit_vectors_rad[unit_vectors_rad <= start_rad] += np.pi * 2
        i = np.argmin(unit_vectors_rad) - len(unit_vectors_rad)
        for c in range(len(unit_vectors_rad)):
            while unit_vectors_rad[i] > end_rad and (k - k0) < len(hull_vertices):
                k += 1
                start_rad = hull_vertices_rad[k]
                end_rad = hull_vertices_rad[k + 1]
                if start_rad > end_rad:
                    end_rad = end_rad + np.pi * 2
                temp_eq = hull_vertices[k + 1] - hull_vertices[k]
                temp_eq[0], temp_eq[1] = temp_eq[1], -temp_eq[0]
                temp_eq = temp_eq / np.linalg.norm(temp_eq)
                temp_eq = np.append(temp_eq, np.dot(temp_eq, hull_vertices[k]))
            ray_points_on_boundary[i] = np.linalg.solve(\
                                np.array([ray_equations[i, :2], temp_eq[:2]]), \
                                np.array([ray_equations[i, -1], temp_eq[-1]]))
            i += 1

        nbd_of_detect_pts_ids, partition_xy, partition_pitch = nbd_detecter(point_xy, ray_points_on_boundary, detect_R, dim = 2, \
                maximal_possible_detect_R = 0.3, partition_pitch = 0.1, partition_xy = None)
        planes = []
        for i in range(len(nbd_of_detect_pts_ids)):
            temp_xy = point_xy[nbd_of_detect_pts_ids[i], :].copy()
            temp_xy = np.column_stack([temp_xy, np.ones(len(temp_xy))])
            temp_z = point_z[nbd_of_detect_pts_ids[i]]
            # print("temp_xy: ", temp_xy)
            n_ = np.dot(np.linalg.inv(np.dot(temp_xy.T, temp_xy)), \
                    np.dot(temp_xy.T, temp_z[..., None])).reshape(-1)
            planes.append(n_)
        planes = np.array(planes)

        ex_point_xy = []
        ex_point_z = []
        for i in range(len(ray_points_on_boundary)):
            temp_xy = np.linspace(0, extend_distance, 1 + int(15 * extend_density))
            temp_xy = np.tile(unit_vectors[i], (len(temp_xy), 1)) * temp_xy[..., None]
            temp_xy = temp_xy + ray_points_on_boundary[i]
            temp_z = np.dot(planes[i].reshape(1, 3), np.row_stack([temp_xy.T, np.ones(len(temp_xy))])).reshape(-1)
            ex_point_xy.append(temp_xy)
            ex_point_z.append(temp_z)
        ex_point_xy = np.row_stack(ex_point_xy)
        ex_point_z = np.concatenate(ex_point_z)
        return ex_point_xy, ex_point_z, partition_xy, partition_pitch

def nbd_detecter(pts, detect_pts, detect_R, dim = 2, \
    maximal_possible_detect_R = 0.3, partition_pitch = 0.1, partition_xy = None):
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

        # partition_pitch = 0.1
        detect_min_x = np.min(detect_pts[:, 0])
        detect_total_distance_x = (np.max(detect_pts[:, 0]) - detect_min_x)
        min_x = np.min(pts[:, 0])
        total_distance_x = (np.max(pts[:, 0]) - min_x)
        total_partition_x = int(total_distance_x / partition_pitch) if total_distance_x > partition_pitch else 1
        partition_pitch_x = total_distance_x / total_partition_x
        number_of_min_side_extended_x = int(max(min_x - detect_min_x, 0) // partition_pitch_x + \
                                            maximal_possible_detect_R // partition_pitch_x + 10)
        number_of_max_side_extended_x = int(max((detect_min_x + detect_total_distance_x) - \
                                            (min_x + total_distance_x), 0) // partition_pitch_x + \
                                            maximal_possible_detect_R // partition_pitch_x + 10)
        number_of_min_side_extended_x = 0
        number_of_max_side_extended_x = 0

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
        number_of_min_side_extended_y = 0
        number_of_max_side_extended_y = 0

        if partition_xy is None:
            partition_x = []
            temp = np.arange(len(pts))
            for i in range(total_partition_x):
                TF = pts[temp, 0] <= partition_pitch_x * (i + 1) + min_x
                partition_x.append(temp[TF])
                temp = temp[~TF]
            partition_x = [np.empty(0)] * number_of_min_side_extended_x + \
                                                partition_x + \
                                                [np.empty(0)] * number_of_max_side_extended_x

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
            # print("temp_xy in nbd: ", temp_xy)
            temp_xy = list(temp_xy.reshape(-1))
            if len(temp_xy) > 0:
                temp_xy = np.concatenate(temp_xy).astype(int)
            else:
                temp_xy = np.empty(0)
            nbd_of_detect_pts_ids.append(temp_xy)
            if len(nbd_of_detect_pts_ids[-1]):
                temp_ = pts[nbd_of_detect_pts_ids[-1], :] - detect_pts[i]
                nbd_of_detect_pts_ids[-1] = nbd_of_detect_pts_ids[-1][np.linalg.norm(temp_, axis = 1) < detect_R]

            # temp = pts - detect_pts[i]
            # temp = pts[np.linalg.norm(temp, axis = 1) < detect_R, :]
            # print(time.time() - s_time)

        print("\nALL:", time.time() - start_time)
        return nbd_of_detect_pts_ids, partition_xy, partition_pitch

