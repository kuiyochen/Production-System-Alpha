'''
The new mold designed function next round would be T(F^), where T is the plastic_mold_transform operator that map the plastic surface to the mold designed function. EX: T(f)(x, y) := s f(x / s, y / s) is the linear enlargement operator.

Use "reader.py" to read the measure data to a numpy array, 
then by data extension and smoothing calculator and 2D interpolation in "calculate.py"
we get an approximated 2D function M(x, y).

For mold, define a new surface as follow ~
Let F(x, y) be the mold surface(last time). The new mold surface function F^(x, y) = F(x, y) - M(x, y).
When you measure the new mold, use the mold designed function this round.

For plastic, define a new surface of new mold as follow ~
Let F(x, y) be the plastic surface(last time). The new plastic surface function F^(x, y) = F(x, y) - M(x, y).
When you measure the new plastic, use the plastic designed function.
Notice that there's some tricks when processing the measurement of the plastic. We may recalculate the errors by choosing a good new plastic_mold_transform operator T' such that the errors are as small as possible before we smoothing the errors. The smoothing process of the plastic must use at "the last time" when the errors are truly truly small enough. Ex: T' can take linear enlargement operator with different "s".

Just use _g_code_ generator in "calculate.py" with mold surface function and varies parameters as input to make what u want.
'''

from reader import *
from calculate import *
def Lens_Mold_transform_operator(func, grad_func, Hess_func):
    new_func = func
    new_grad_func = grad_func
    new_Hess_func = Hess_func
    interpolation = True
    if interpolation:
        new_grad_func = lambda x, y: interpolation_grad_func(x, y, new_func, epsilon = 10**-10, form = "stack_array")
        new_Hess_func = lambda x, y: interpolation_Hess_func(x, y, new_func, epsilon = 10**-10, form = "stack_array")
    return new_func, new_grad_func, new_Hess_func


source = "XX.csv"
partition_data = None
detect_R = 4.
smoothness_R = 4.
smoothness_grid_pitch = 0.7
boundary = "auto"
reference_of_zero_correction_point = np.array([0., 0.])
UA3P_xy, UA3P_z, UA3P_zd, dim, _ = UA3P_reader(source = measure_source, output_column_stack = "xy")
assert dim == 3
ex_point_xy, ex_point_z, partition_data = data_extended(UA3P_xy, UA3P_zd, detect_R, 
    boundary = boundary, extended_mode = extended_mode, 
    extend_distance = extend_distance, extend_density = extend_density, partition_data = partition_data)
extended_mode = {"xy_shape": "normal to boundary", # "circle"
                "centering": np.array([0., 0.]), # "auto", only used when "xy_shape" is "circle"
                "extend_radius": 10., 
                "clipped": np.array([-20., 20]), 
                "back_to_zero_extend_radius": 10., # None if don't want it.
                "extend_density": 1., 
                }
ex_point_xy, ex_point_z, z_shift, partition_data = data_extended(point_xy, point_z, detect_R, 
    smoothness_R, 
    smoothness_grid_pitch, 
    reference_of_zero_correction_point = reference_of_zero_correction_point, 
    boundary = boundary, 
    extended_mode = extended_mode, 
    partition_data = partition_data)
'''
data_extended classification
given reference_of_zero_correction_point(no option, default by (0., 0.))
                (need smoothness_R)
xy:
    ((auto/manual) centering and extend to circle/normal to boundary) 
                                            with (auto/manual) extend_radius
z(after extend by gradient):
    clipped by given ceiling and floor(Yes/No)
    back to zero(Yes/No)
            xy: with back_to_zero_extend_radius

extended_mode = {"xy_shape": "circle", # "normal to boundary"
                "centering": "auto", # "manual", only used when "xy_shape" is "circle"
                "extend_radius": 2., 
                "clipped": np.array([np.inf, 0.001]), 
                "back_to_zero_extend_radius": 1., # None if don't want it.
                }

'''

measurement_center_in_func_coordinate = np.array([0., 0.])
UA3P_xy = UA3P_xy
ex_point_xy = ex_point_xy
UA3P_z = UA3P_z
ex_point_z = ex_point_z
pts = np.row_stack([UA3P_xy, ex_point_xy])
z = np.concatenate([UA3P_z, ex_point_z])
interpolation_data = smooth_calculator(pts, z, smoothness_R, smoothness_grid_pitch, partition_data = None, print_out = False)
correction_func = lambda x, y: interpolation_2D(x, y, interpolation_data)
correction_func_grad = lambda x, y: interpolation_grad_func(x, y, correction_func, epsilon = 10**-10, form = "stack_array")

contact_points, lead_in_process_contact_points_in_func_coordinate = \
get_gcode(contact_points_in_func_coordinate, \
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
    )



