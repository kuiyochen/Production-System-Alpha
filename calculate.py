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

        nbd_of_detect_pts_ids, partition_data = nbd_detecter(point_xy, ray_points_on_boundary, detect_R, dim = 2, \
                partition_pitch = 0.1, partition_data = None)
        planes = []
        for i in range(len(nbd_of_detect_pts_ids)):
            temp_xy = point_xy[nbd_of_detect_pts_ids[i], :].copy()
            temp_xy = np.column_stack([temp_xy, np.ones(len(temp_xy))])
            temp_z = point_z[nbd_of_detect_pts_ids[i]]
            try:
                n_ = np.dot(np.linalg.inv(np.dot(temp_xy.T, temp_xy)), \
                        np.dot(temp_xy.T, temp_z[..., None])).reshape(-1)
            except Exception as e:
                if "singular" in str(e).lower():
                    raise TypeError("Singular Matrix!\n"\
                         + " detect_R so small that there's no enough data for prediction.\n"\
                         + f"number of data: {len(temp_xy)}")
                else:
                    raise e
            planes.append(n_)
        planes = np.array(planes)

        ex_point_xy = []
        ex_point_z = []
        temp_norms = np.linspace(0, extend_distance, 1 + int(15 * extend_density))[..., None]
        for i in range(len(ray_points_on_boundary)):
            temp_xy = np.tile(unit_vectors[i], (len(temp_norms), 1)) * temp_norms \
                             + ray_points_on_boundary[i]
            temp_z = np.dot(planes[i].reshape(1, 3), np.row_stack([temp_xy.T, np.ones(len(temp_xy))])).reshape(-1)
            ex_point_xy.append(temp_xy)
            ex_point_z.append(temp_z)
        ex_point_xy = np.row_stack(ex_point_xy)
        ex_point_z = np.concatenate(ex_point_z)
        return ex_point_xy, ex_point_z, partition_data

def nbd_detecter(pts, detect_pts, detect_R, dim = 2, \
    partition_pitch = 0.1, partition_data = None):
    if dim == 2:
        import time
        start_time = time.time()

        # pts = (np.random.rand(10**4 * 2, 2) * 2 - 1)
        # detect_pts = (np.random.rand((2 * 100)**2, 2) * 2 - 1) * 1.2
        # # maximal_possible_detect_R = 0.3
        # detect_R = 0.2
        # partition_pitch = 0.1
        # partition_data = None

        pp = np.linspace(0, 100.1, len(detect_pts)).astype(int)
        filter_pp = np.diff(np.append(pp, 100)) > 0
        filter_pp[-1] = True

        detect_min_x = np.min(detect_pts[:, 0])
        detect_total_distance_x = (np.max(detect_pts[:, 0]) - detect_min_x)
        detect_min_y = np.min(detect_pts[:, 0])
        detect_total_distance_y = (np.max(detect_pts[:, 0]) - detect_min_y)

        if partition_data is None:
            min_x = np.min(pts[:, 0])
            total_distance_x = (np.max(pts[:, 0]) - min_x)
            total_partition_x = int(total_distance_x / partition_pitch) if total_distance_x > partition_pitch else 1
            partition_pitch_x = total_distance_x / total_partition_x
            min_y = np.min(pts[:, 1])
            total_distance_y = (np.max(pts[:, 1]) - min_y)
            total_partition_y = int(total_distance_y / partition_pitch) if total_distance_y > partition_pitch else 1
            partition_pitch_y = total_distance_y / total_partition_y
            partition_x = []
            temp = np.arange(len(pts))
            for i in range(total_partition_x):
                TF = pts[temp, 0] <= partition_pitch_x * (i + 1) + min_x
                partition_x.append(temp[TF])
                temp = temp[~TF]
            partition_xy = np.array([[0] * total_partition_y] * total_partition_x, dtype = object)
            for i in range(total_partition_x):
                temp = partition_x[i].copy()
                for j in range(total_partition_y):
                    TF = pts[temp, 1] <= partition_pitch_y * (j + 1) + min_y
                    partition_xy[i, j] = temp[TF].copy()
                    temp = temp[~TF]
            partition_data = {"partition_xy": partition_xy, 
                          "partition_pitch": partition_pitch, 
                          "min_x": min_x, 
                          "total_distance_x": total_distance_x, 
                          "total_partition_x": total_partition_x, 
                          "partition_pitch_x": partition_pitch_x, 
                          "min_y": min_y, 
                          "total_distance_y": total_distance_y, 
                          "total_partition_y": total_partition_y, 
                          "partition_pitch_y": partition_pitch_y, 
                         }
        else:
            partition_xy = partition_data["partition_xy"].copy()
            partition_pitch = partition_data["partition_pitch"]
            min_x = partition_data["min_x"]
            total_distance_x = partition_data["total_distance_x"]
            total_partition_x = partition_data["total_partition_x"]
            partition_pitch_x = partition_data["partition_pitch_x"]
            min_y = partition_data["min_y"]
            total_distance_y = partition_data["total_distance_y"]
            total_partition_y = partition_data["total_partition_y"]
            partition_pitch_y = partition_data["partition_pitch_y"]

        temp = (int(2 * detect_R / partition_pitch) + 1)
        nbd_partition_id = (temp + 1) // 2 if temp % 2 else temp // 2
        nbd_of_detect_pts_ids = []
        print("\nprepare time:", time.time() - start_time)
        for i in range(len(detect_pts)):
            if filter_pp[i]:
                print(pp[i], end = "\r")
            partition_id_x = int((detect_pts[i, 0] - min_x) / partition_pitch_x)
            partition_id_y = int((detect_pts[i, 1] - min_y) / partition_pitch_y)

            temp_xy = partition_xy[max(partition_id_x - nbd_partition_id, 0):\
                                   min(partition_id_x + nbd_partition_id + 1, total_partition_x), 
                                    max(partition_id_y - nbd_partition_id, 0):\
                                   min(partition_id_y + nbd_partition_id + 1, total_partition_y)]

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
            # print("nbd time: ", time.time() - s_time)

        print("\nALL:", time.time() - start_time)
        return nbd_of_detect_pts_ids, partition_data

def get_gcode(contact_points_in_func_coordinate, \
    func, \
    grad, \
    ncp_name = "test_path.ncp", \
    init_func_center_in_machine_coordinate = np.zeros(2), \
    # init_func_x_axis_angle_in_machine_coordinate = 0., \
    gradient_of_origin_in_machine_coordinate = np.zeros(2), \
    func_center_height_in_machine_coordinate = 0., \
    tool_R = 0.05, \
    lathe_tilt = 0., \
    lathe_fix_swing = 0., \
    lead_out_process_angles_in_func_coordinate = np.arange(0, 360 * 4, 1.) % 360, \
    lead_out_left = 0.15, \
    lead_in_process_contact_points_in_func_coordinate = None, \
    lead_in_left = 0.05, \
    B_axis = "OFF", \
    B_bound = 80., \
    B_mitigate_rate_bound = 0.9, \
    B_scale_mitigate_rate = 1.5, \
    lathe_C = "OFF", \
    lathe_C_mitigate_rate = 2.0, \
    Counterclockwise = False, \
    F_axis = "OFF", \
    F_max = 9000, \
    F_min = 10, \
    F_mitigate_rate = 0.8
    ):
    '''
        Naturally, lathe_tilt <= 0
        Counterclockwise False for ULC True for PT
        lathe_C would open the Y axis
        lathe_fix_swing means the swing angle in lead out, it's fix.
    '''
    B_axis = B_axis.upper()
    lathe_C = lathe_C.upper()
    F_axis = F_axis.upper()
    assert B_axis in ["ON", "OFF"], ""
    assert lathe_C in ["ON", "OFF"], ""
    assert F_axis in ["ON", "OFF"], ""
    assert F_max > F_min, ""
    gcode_axis_name = ["C", "X", "Z", "Y", "B", "F"]
    if F_axis == "OFF":
        del gcode_axis_name[-1]
        if B_axis == "OFF":
            del gcode_axis_name[-1]
    elif B_axis == "OFF":
        del gcode_axis_name[-2]
    if lathe_C == "OFF" and \
        ((init_func_center_in_machine_coordinate == np.zeros(2)).all() and \
        lathe_tilt == 0.):
        del gcode_axis_name[3]

    # s = np.sin(np.deg2rad(init_func_x_axis_angle_in_machine_coordinate))
    # c = np.cos(np.deg2rad(init_func_x_axis_angle_in_machine_coordinate))
    # init_func_center_in_machine_coordinate = np.dot(\
    #     np.array([[c, s], [-s, c]]), init_func_center_in_machine_coordinate)

    contact_points = contact_points_in_func_coordinate.copy()
    i = -1
    while (contact_points[i] == np.zeros(2)).all():
        i -= 1
    contact_points = np.row_stack([contact_points[:i + 1 if i < -1 else len(contact_points)], np.zeros((1, 2))])
    if lead_in_process_contact_points_in_func_coordinate is not None:
        contact_points = np.row_stack([lead_in_process_contact_points_in_func_coordinate, contact_points])

    lathe_counterclockwise = ((np.cross(contact_points[:-1], contact_points[1:], axis = 1)) >= 0).all()
    print("lathe_counterclockwise", lathe_counterclockwise)

    if type(func) != type(np.array([0.])):
        func = func(contact_points)
    if type(grad) != type(np.array([0.])):
        center_grad = grad(np.zeros((len(lead_out_process_angles_in_func_coordinate), 2)))
        grad = grad(contact_points)

    def normalized(v):
        if len(v.shape) == 1:
            return v / np.linalg.norm(v)
        return v / np.linalg.norm(v, axis = -1)[..., None]
    print("calculate gradient ...")
    surf_n = normalized(np.column_stack([-grad, np.ones(len(grad))]))
    origin_surf_n = normalized(np.append(-gradient_of_origin_in_machine_coordinate, 1))

    if not B_axis == "OFF":
        print("calculate B_array_temp ...")
        B_array_temp = np.einsum("ij,ij->i", grad[:-1], normalized(contact_points)[:-1])
        B_array_temp = np.rad2deg(np.arctan(np.append(B_array_temp, B_array_temp[-1])))
        if B_axis == "ON":
            max_abs_B = np.max(np.abs(B_array_temp))
            if max_abs_B > 0:
                B_scale = min(B_bound / max_abs_B, B_mitigate_rate_bound)
                B_array_temp = B_array_temp * B_scale
                B_scale_mitigate = (np.linspace(0., 1., len(B_array_temp))**B_scale_mitigate_rate)[::-1]
                B_array_temp = B_array_temp * B_scale_mitigate
                del B_scale_mitigate
                gc.collect()
    else:
        B_array_temp = np.full(len(contact_points), lathe_fix_swing)

    print("calculate dpn ...")
    p = contact_points[:-1].copy()
    pn = normalized(p)
    pnp = pn[:, [1, 0]].copy()
    pnp[:, 0 if lathe_counterclockwise else 1] *= -1
    if not lathe_C == "OFF":
        dp = contact_points[1:, :] - contact_points[:-1, :]
        dpn = normalized(dp)
        if lathe_C == "ON":
            lathe_C_mitigate_interpolation = (np.linspace(0., 1., len(dp))**lathe_C_mitigate_rate)[..., None][::-1]
            dpn = normalized((1 - lathe_C_mitigate_interpolation) * pnp + lathe_C_mitigate_interpolation * dpn)
            del lathe_C_mitigate_interpolation
            gc.collect()
    else:
        dpn = pnp.copy()
    p = contact_points.copy()
    temp = np.deg2rad(lead_out_process_angles_in_func_coordinate[0])
    temp = np.array([-np.sin(temp), np.cos(temp)]) * (1 if lathe_counterclockwise else -1)
    dpn = np.row_stack([dpn, temp.reshape(-1, 2)])
    dpnp = dpn[:, [1, 0]].copy()
    dpnp[:, 1 if lathe_counterclockwise else 0] *= -1

    print("lead out process settings ...")
    lead_out_C_array_temp = lead_out_process_angles_in_func_coordinate
    lead_out_B_array_temp = np.full_like(lead_out_C_array_temp, B_array_temp[-1])
    th = np.deg2rad(lead_out_C_array_temp)
    lead_out_dpnp = np.array([np.cos(th), np.sin(th)]).T
    lead_out_dpn = lead_out_dpnp[:, [1, 0]].copy()
    lead_out_dpn[:, 0 if lathe_counterclockwise else 1] *= -1
    lead_out_surf_n = normalized(np.column_stack([-center_grad, np.ones(len(lead_out_C_array_temp))]))
    del th
    gc.collect()

    print("calculate tool face normal ...")
    temp = np.deg2rad(lathe_tilt)
    sin_tilt, cos_tilt = np.sin(temp), np.cos(temp)
    temp = np.deg2rad(B_array_temp)
    sin_swing, cos_swing = np.sin(temp), np.cos(temp)
    temp = np.deg2rad(lathe_fix_swing)
    s, c = np.sin(temp), np.cos(temp)
    temp_zeros = np.zeros(len(sin_swing))
    temp_ones = np.ones(len(sin_swing))
    Rotation_swing_matrices = np.transpose(np.array([
                                [cos_swing, temp_zeros, -sin_swing], 
                                [temp_zeros, temp_ones, temp_zeros], 
                                [sin_swing, temp_zeros, cos_swing]
                                ]), [2, 0, 1])
    basis_changing_matrices = np.transpose(np.array([
                                [dpnp[:, 0], dpn[:, 0], temp_zeros], 
                                [dpnp[:, 1], dpn[:, 1], temp_zeros], 
                                [temp_zeros, temp_zeros, temp_ones]
                                ]), [2, 0, 1])
    Rotation_swing_matrices = np.einsum("ijk,ikl->ijl", Rotation_swing_matrices, basis_changing_matrices)
    Rotation_swing_matrices = np.einsum("ijk,ilk->ijl", basis_changing_matrices, Rotation_swing_matrices)
    tool_face_normal = np.einsum("ikj,ij->ik", Rotation_swing_matrices, 
                                            np.column_stack([cos_tilt * dpn, np.full(len(dpn), sin_tilt)]))
    origin_tool_face_normal = np.dot(np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]), 
                                    np.array([0, cos_tilt, sin_tilt]))
    del Rotation_swing_matrices, basis_changing_matrices, sin_swing, cos_swing, s, c, temp_zeros, temp_ones, 
    gc.collect()

    print("calculate lead out tool face normal ...")
    temp = np.deg2rad(lead_out_B_array_temp)
    lead_out_sin_swing, lead_out_cos_swing = np.sin(temp), np.cos(temp)
    lead_out_temp_zeros = np.zeros(len(lead_out_sin_swing))
    lead_out_temp_ones = np.ones(len(lead_out_sin_swing))
    lead_out_Rotation_swing_matrices = np.transpose(np.array([
                                [lead_out_cos_swing, lead_out_temp_zeros, -lead_out_sin_swing], 
                                [lead_out_temp_zeros, lead_out_temp_ones, lead_out_temp_zeros], 
                                [lead_out_sin_swing, lead_out_temp_zeros, lead_out_cos_swing]
                                ]), [2, 0, 1])
    lead_out_basis_changing_matrices = np.transpose(np.array([
                                [lead_out_dpnp[:, 0], lead_out_dpn[:, 0], lead_out_temp_zeros], 
                                [lead_out_dpnp[:, 1], lead_out_dpn[:, 1], lead_out_temp_zeros], 
                                [lead_out_temp_zeros, lead_out_temp_zeros, lead_out_temp_ones]
                                ]), [2, 0, 1])
    lead_out_Rotation_swing_matrices = np.einsum("ijk,ikl->ijl", lead_out_Rotation_swing_matrices, lead_out_basis_changing_matrices)
    lead_out_Rotation_swing_matrices = np.einsum("ijk,ilk->ijl", lead_out_basis_changing_matrices, lead_out_Rotation_swing_matrices)
    lead_out_tool_face_normal = np.einsum("ikj,ij->ik", lead_out_Rotation_swing_matrices, 
                                            np.column_stack([cos_tilt * lead_out_dpn, np.full(len(lead_out_dpn), sin_tilt)]))
    del lead_out_sin_swing, lead_out_cos_swing, lead_out_temp_zeros, lead_out_temp_ones, lead_out_Rotation_swing_matrices, lead_out_basis_changing_matrices
    gc.collect()

    print("calculate relative offset ...")
    relative_offset = normalized(surf_n - np.einsum("ij,ij->i", surf_n, tool_face_normal)[:, None] * tool_face_normal) * tool_R
    origin_offset = normalized(origin_surf_n - np.dot(origin_surf_n, origin_tool_face_normal) * origin_tool_face_normal) * tool_R
    tuning_offset = normalized(np.array([0, 0, 1.]) - np.dot(np.array([0, 0, 1.]), origin_tool_face_normal) * origin_tool_face_normal) * tool_R
    del surf_n, tool_face_normal
    gc.collect()
    lead_out_relative_offset = normalized(lead_out_surf_n - np.einsum("ij,ij->i", lead_out_surf_n, lead_out_tool_face_normal)[:, None] * lead_out_tool_face_normal) * tool_R
    del lead_out_surf_n, lead_out_tool_face_normal
    gc.collect()

    print("tuning C_array_temp")
    C_array_temp = np.rad2deg(np.arctan2(p[:, 1], p[:, 0]))
    if not lathe_C == "OFF":
        arccos_ = np.append(np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i", dpnp[:-1], pn), -1., 1.))), 0.)
        C_array_temp = C_array_temp + (arccos_ * 1 if lathe_counterclockwise else -1)
    if not B_axis == "OFF":
        tilt_swing_C = np.abs(np.rad2deg(np.arctan(-np.sin(np.deg2rad(B_array_temp)) * sin_tilt / cos_tilt)))\
                     * np.sign(B_array_temp)
        C_array_temp = C_array_temp + tilt_swing_C
        lead_out_tilt_swing_C = np.abs(np.rad2deg(np.arctan(-np.sin(np.deg2rad(lead_out_B_array_temp)) * sin_tilt / cos_tilt)))\
                     * np.sign(lead_out_B_array_temp)
        lead_out_C_array_temp = lead_out_C_array_temp + lead_out_tilt_swing_C
        del tilt_swing_C, lead_out_tilt_swing_C
        gc.collect()

    print("calculate arrays ...")
    if not B_axis == "closed":
        B_array = B_array_temp.copy()
        B_array = np.concatenate([B_array_temp, lead_out_B_array_temp[1:]])
    C_array_temp = C_array_temp % 360
    temp = np.deg2rad(C_array_temp)
    sin_C, cos_C = np.sin(temp), np.cos(temp)
    temp_zeros = np.zeros(len(sin_C))
    temp_ones = np.ones(len(sin_C))
    temp_points_ = np.einsum("ij,ijk->ik", 
        np.column_stack([p + init_func_center_in_machine_coordinate, \
            func + func_center_height_in_machine_coordinate]) + relative_offset, \
        np.transpose(np.array([
            [cos_C, -sin_C, temp_zeros], 
            [sin_C, cos_C, temp_zeros], 
            [temp_zeros, temp_zeros, temp_ones]
            ]), [2, 0, 1])
        )
    lead_out_C_array_temp = lead_out_C_array_temp % 360
    temp = np.deg2rad(lead_out_C_array_temp)
    lead_out_sin_C, lead_out_cos_C = np.sin(temp), np.cos(temp)
    lead_out_temp_zeros = np.zeros(len(lead_out_sin_C))
    lead_out_temp_ones = np.ones(len(lead_out_sin_C))
    lead_out_temp_points_ = np.einsum("ij,ijk->ik", 
        np.column_stack([np.zeros((len(lead_out_sin_C), 2)) + init_func_center_in_machine_coordinate, \
            np.full(len(lead_out_sin_C), func[-1]) + func_center_height_in_machine_coordinate]) + lead_out_relative_offset, \
        np.transpose(np.array([
            [lead_out_cos_C, -lead_out_sin_C, lead_out_temp_zeros], 
            [lead_out_sin_C, lead_out_cos_C, lead_out_temp_zeros], 
            [lead_out_temp_zeros, lead_out_temp_zeros, lead_out_temp_ones]
            ]), [2, 0, 1])
        )
    lead_out_temp_points_[:, 2] += np.linspace(0, lead_out_left, len(lead_out_temp_points_))

    F_array_temp = (np.linspace(F_min / F_max, 1., len(C_array_temp))**F_mitigate_rate)[::-1] * F_max
    lead_out_F_array_temp = (np.linspace(F_min / F_max, 1., len(lead_out_C_array_temp))**(0.5)) * F_max
    F_array_temp = np.concatenate([F_array_temp, lead_out_F_array_temp[1:]])
    C_array_temp = np.concatenate([C_array_temp, lead_out_C_array_temp[1:]])
    temp_points_ = np.row_stack([temp_points_, lead_out_temp_points_[1:]])

    F_array = F_array_temp.copy()
    C_array = C_array_temp % 360
    if not Counterclockwise:
        C_array *= -1
    temp = tuning_offset - origin_offset
    print("對刀偏移量：", -temp[:2])
    X_array = temp_points_[:, 0] - tuning_offset[0]
    Y_array = temp_points_[:, 1] - tuning_offset[1]
    Z_array = temp_points_[:, 2] - origin_offset[2]

    if lead_in_process_contact_points_in_func_coordinate is not None:
        temp_len = len(lead_in_process_contact_points_in_func_coordinate)
        temp = (np.linspace(0., lead_in_left**(0.5), temp_len)**2)[::-1]
        Z_array[:temp_len] = Z_array[:temp_len] + temp

    arrays = []
    for name in gcode_axis_name:
        exec("arrays.append(" + name + "_array)", globals(), locals())
    arrays = np.column_stack(arrays)
    print("gcode writer ...")
    gcode_writer(ncp_name, gcode_axis_name, arrays)
    if lead_in_process_contact_points_in_func_coordinate is not None:
        return contact_points[:len(lead_in_process_contact_points_in_func_coordinate)], lead_in_process_contact_points_in_func_coordinate
    else:
        return contact_points, None

def gcode_writer(ncp_name, gcode_axis_name, arrays):
    F_axis = "F" in gcode_axis_name
    gcode_axis_name_array = np.tile(gcode_axis_name, (len(arrays), 1))
    gcode_axis_name_array = np.column_stack([gcode_axis_name_array, ["\n"] * len(gcode_axis_name_array)])
    if not F_axis:
        str_arrays = (np.vectorize(lambda v: f"{v:.7f}"))(arrays)
    else:
        str_arrays = (np.vectorize(lambda v: f"{v:.7f}"))(arrays[:, :-1])
        str_F_array = (np.vectorize(lambda v: str(int(v))))(arrays[:, -1])
        str_arrays = np.column_stack([str_arrays, str_F_array])
    str_arrays = np.column_stack([str_arrays, [""] * len(str_arrays)])
    str_ = "".join(
        np.concatenate([gcode_axis_name_array[..., None], str_arrays[..., None]], \
        axis = -1).reshape(-1))
    with open(ncp_name, "w") as f:
        f.write(str_)
    f.close()
    return str_

# pitch = 0.2
# degree_pitch = 5
# max_radius = 1.5
# assert 360 % degree_pitch == 0., ""
# ths = np.arange(0, 360 * (1 + max_radius // pitch), degree_pitch) % 360
# radiuses = np.linspace(0., max_radius, len(ths))[::-1]
# temp = radiuses * np.exp(np.deg2rad(ths) * 1J)
# contact_points_in_func_coordinate = np.column_stack([temp.real, temp.imag])
# print(contact_points_in_func_coordinate)
# ths = np.arange(0, 360 * 4, degree_pitch) % 360
# temp = max_radius * np.exp(np.deg2rad(ths) * 1J)
# lead_in_process_contact_points_in_func_coordinate = np.column_stack([temp.real, temp.imag])
# print(lead_in_process_contact_points_in_func_coordinate)
# func = lambda points: 0.5 * points[:, 0]**2 + 0.01 * points[:, 1]**2
# grad = lambda points: np.column_stack([0.5 * points[:, 0] * 2, 0.01 * points[:, 1] * 2])
# get_gcode(contact_points_in_func_coordinate = contact_points_in_func_coordinate, \
#             func = func, \
#             grad = grad, \
#             ncp_name = "test_path.ncp", \
#             init_func_center_in_machine_coordinate = np.array([0., 0.]), \
#             # init_func_x_axis_angle_in_machine_coordinate = 60., \
#             gradient_of_origin_in_machine_coordinate = np.array([0., 0.]), \
#             func_center_height_in_machine_coordinate = 0., \
#             tool_R = 0.05, \
#             lathe_tilt = 0., \
#             lathe_fix_swing = 0., \
#             lead_out_process_angles_in_func_coordinate = np.arange(0, 360 * 4, 1) % 360, \
#             lead_out_left = 0.15, \
#             lead_in_process_contact_points_in_func_coordinate = lead_in_process_contact_points_in_func_coordinate, \
#             lead_in_left = 0.05, \
#             B_axis = "OFF", \
#             B_bound = 80., \
#             B_mitigate_rate_bound = 0.9, \
#             B_scale_mitigate_rate = 1.5, \
#             lathe_C = "OFF", \
#             lathe_C_mitigate_rate = 2.0, \
#             Counterclockwise = True, \
#             F_axis = "OFF", \
#             F_max = 9000, \
#             F_min = 10, \
#             F_mitigate_rate = 0.8
#             )

def interpolation_2D(x, y, interpolation_data = {"x_start_end_pitch": (-1., 1., 1.), 
    "y_start_end_pitch": (-1., 1., 1.), 
    "grid_shape": (3, 3), 
    "grid_x": np.array([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]), 
    "grid_y": np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]), 
    "grid_z": np.zeros((3, 3)).astype(np.float64)}):
    x_start, x_end, x_pitch = interpolation_data["x_start_end_pitch"]
    y_start, y_end, y_pitch = interpolation_data["y_start_end_pitch"]
    grid_shape = interpolation_data["grid_shape"]
    grid_x = interpolation_data["grid_x"]
    grid_y = interpolation_data["grid_y"]
    grid_z = interpolation_data["grid_z"].copy()
    grid_z = np.column_stack([np.zeros(len(grid_z)), grid_z, np.zeros(len(grid_z))])
    grid_z = np.row_stack([np.zeros(grid_z.shape[1]), grid_z, np.zeros(grid_z.shape[1])])

    id_x, res_x = divmod((x - x_start), x_pitch)
    id_y, res_y = divmod((y - y_start), y_pitch)
    res_x = res_x / x_pitch
    res_y = res_y / y_pitch
    temp_id_x = id_x.copy()
    temp_id_y = id_y.copy()
    temp_id_x = np.clip(temp_id_x + 1, 0, grid_shape[0]).astype(int)
    temp_id_y = np.clip(temp_id_y + 1, 0, grid_shape[1]).astype(int)
    temp_x0 = grid_z[temp_id_x, temp_id_y] * (1 - res_y) + grid_z[temp_id_x, temp_id_y + 1] * res_y
    temp_x1 = grid_z[temp_id_x + 1, temp_id_y] * (1 - res_y) + grid_z[temp_id_x + 1, temp_id_y + 1] * res_y
    z = temp_x0 * (1 - res_x) + temp_x1 * res_x
    z[id_x < -1] = 0.
    z[id_x >= grid_shape[0]] = 0.
    z[id_y < -1] = 0.
    z[id_y >= grid_shape[1]] = 0.
    return z

