# http://www.poketcode.com/en/pyglet_demos/index.html
import os
import numpy as np
from colorsys import hsv_to_rgb

import pyglet
from pyglet.gl import *
import ctypes
from calculate import *


class Window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_init_setting = {"x": 0, "y":0, "z":-150, "rx":-69.4, "ry":0, "rz":24.6}
        self.camera = GL_camera(self.camera_init_setting)

        self.mouse_pos = (0, 0)
        self.mv_mat = (GLdouble * 16)()
        self.p_mat  = (GLdouble * 16)()
        self.v_rect = (GLint * 4)()


        # config = pyglet.gl.Config(sample_buffers=1, samples=4)
        # window = pyglet.window.Window(width=width, height=height, config=config, caption='GL_Plot', resizable=True)
        # window = pyglet.window.Window(width=width, height=height, caption='GL_Plot', resizable=True)
        self.set_location((screensize[0] - self.width) // 2, (screensize[1] - self.height) // 2)
        cursor = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(cursor)
        # backgroundcolor
        glClearColor(0.95, 0.95, 0.95, 1)

        glEnable(GL_DEPTH_TEST)
        # glDisable(GL_DEPTH_TEST)
        # glDepthFunc(GL_NEVER)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45., 1., 0.1, 800.)
        glMatrixMode(GL_MODELVIEW)

        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_CULL_FACE)
        glDisable(GL_CULL_FACE)

        self.GL_objs = []
        self.GL_objs.append(Point(self, np.zeros((1, 3)), 3))

        x = np.arange(-100., 100. + 10**-10, 1.3) + 30
        y = np.arange(-100., 100. + 10**-10, 1.3) + 30
        mx, my = np.meshgrid(x, y, indexing='ij')
        # mz = (mx**2 +my**2) / (5 * 10**2) + 10
        mz = np.cos(2 * np.pi * mx/70) * np.cos(2 * np.pi * my/88) * 10 + np.random.randn(*mx.shape)*2
        # mz = (mx**2 + my**2 - 20 * mx - 40 * my) / (3*10**2)
        # mz = (mx**2 + my**2 - 20 * mx - 40 * my) / (5 * 10**2) + np.random.randn(*mx.shape)*2
        self.point_xy = np.column_stack([mx.reshape(-1), my.reshape(-1)])
        self.point_z = mz.reshape(-1)
        extended_mode = {"xy_shape": "normal to boundary", # "circle"
                "centering": np.array([0., 0.]), # "auto", only used when "xy_shape" is "circle"
                "extend_radius": 10., 
                "clipped": np.array([-20., 20]), 
                "back_to_zero_extend_radius": 10., # None if don't want it.
                "extend_density": 1., 
                }
        smoothness_R = 10.
        smoothness_grid_pitch = 1.
        ex_point_xy, ex_point_z, z_shift, partition_data = data_extended(self.point_xy, self.point_z, detect_R = 15, 
                smoothness_R = smoothness_R, 
                smoothness_grid_pitch = smoothness_grid_pitch, 
                reference_of_zero_correction_point = np.array([0., 0.]), 
                boundary = "auto", 
                extended_mode = extended_mode, 
                partition_data = None, 
                print_out = True)
        self.point_xy = np.row_stack([self.point_xy, ex_point_xy])
        self.point_z = np.concatenate([self.point_z, ex_point_z]) - z_shift

        interpolation_data = smooth_calculator(self.point_xy, \
            self.point_z, smoothness_R = smoothness_R, \
            grid_pitch = smoothness_grid_pitch, partition_data = None, print_out = True)
        self.GL_objs.append(Point(self, np.column_stack([self.point_xy, self.point_z]), 3))

        self.plot_grid()

        self.GL_objs.append(Surface(self, interpolation_data["grid_x"], interpolation_data["grid_y"], \
            interpolation_data["grid_z"], 'b'))

        _, self.partition_data = nbd_detecter(self.point_xy, np.zeros((1, 2)), detect_R = 1, \
        partition_pitch = 5, partition_data = None)

        self.mouse_detect_ON = True

    def plot_grid(self, grid_width = 20, number_of_grid_in_axis = 5, axis_length = 100, z_grid_height = 10.):
        grid_ = grid_width * number_of_grid_in_axis // 2
        temp = np.arange(-grid_, grid_ + 10**-1, grid_width)
        temp_ = np.column_stack([temp, np.full_like(temp, grid_), np.zeros_like(temp)])
        temp__ = temp_.copy()
        temp__[:, 1] = -temp__[:, 1]
        temp_ = np.transpose(np.concatenate([temp_[..., None], temp__[..., None]], axis = -1), (0, 2, 1))
        self.GL_objs.append(Line(self, temp_.reshape(-1, 3), 1))

        temp_ = np.column_stack([np.full_like(temp, grid_), temp, np.zeros_like(temp)])
        temp__ = temp_.copy()
        temp__[:, 0] = -temp__[:, 0]
        temp_ = np.transpose(np.concatenate([temp_[..., None], temp__[..., None]], axis = -1), (0, 2, 1))
        self.GL_objs.append(Line(self, temp_.reshape(-1, 3), 1))

        temp = np.array([10., 10., z_grid_height])
        temp_ = np.tile(temp, (4, 1))
        temp_[1:3, 1] = -temp_[1:3, 1]
        temp_[2:4, 0] = -temp_[2:4, 0]
        temp__ = np.zeros((4, 2, 3))
        temp__[:, 0, :] = temp_.copy()
        temp__[:-1, 1, :] = temp_[1:].copy()
        temp__[-1, 1, :] = temp.copy()
        temp__ = temp__.reshape(-1, 3)
        temp_ = temp__.copy()
        temp__[:, 2] = -temp__[:, 2]
        temp_ = np.row_stack([temp_, temp__])
        self.GL_objs.append(Line(self, temp_.copy(), 1))

        self.GL_objs.append(Line(self, np.array([[0, 0, 0.], [axis_length, 0, 0], \
                                    [0, 0, 0.], [0, axis_length, 0], \
                                    [0, 0, 0.], [0, 0, z_grid_height * 2]]), 3))


    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45., width / height, 0.1, 800.)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        return pyglet.event.EVENT_HANDLED

    def mouse_detect(self):
        if not self.mouse_detect_ON:
            return
        glGetDoublev(GL_MODELVIEW_MATRIX, self.mv_mat)
        glGetDoublev(GL_PROJECTION_MATRIX, self.p_mat)
        glGetIntegerv(GL_VIEWPORT, self.v_rect)

        temp_val = [GLdouble() for _ in range(3)]
        gluUnProject(*self.mouse_pos, 0, self.mv_mat, self.p_mat, self.v_rect, *temp_val)
        mouse_near = np.array([v.value for v in temp_val])
        gluUnProject(*self.mouse_pos, 1, self.mv_mat, self.p_mat, self.v_rect, *temp_val)
        mouse_far = np.array([v.value for v in temp_val])
        t = mouse_near[2] / (mouse_near[2] - mouse_far[2])
        if t >= 0.:
            point_on_plane = mouse_far * t + mouse_near * (1 - t)
            detect_R = (self.camera.z)**2 / 5000
            detect_R = np.clip(detect_R, 3., 30.)
            nbd, _ = nbd_detecter(self.point_xy, point_on_plane[:2].reshape(1, 2), \
                detect_R = detect_R, partition_data = self.partition_data)
            nbd = nbd[0]
        if t >= 0. and len(nbd):
            nbd_xy = self.point_xy[nbd, :].copy()
            nbd_z = self.point_z[nbd].copy()
            self.GL_objs[0] = Point(self, np.column_stack([nbd_xy, nbd_z]), 5, color = "k")
        else:
            self.GL_objs[0] = Point(self, np.zeros((1, 3)), 1)

    def on_draw(self):

        # clears the background with the background color
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # wire-frame mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        self.mouse_detect()

        for obj in self.GL_objs:
            obj.draw()

        glFlush()

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_pos = (x, y)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y): # zoom
        self.mouse_pos = (x, y)
        self.camera.on_mouse_scroll(scroll_x, scroll_y)

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.mouse_pos = (x, y)
        key = pyglet.window.key
        if button == pyglet.window.mouse.MIDDLE:
            self.camera.mouse_MIDDLE(dx, dy)
            return
        if button == pyglet.window.mouse.RIGHT:
            self.camera.mouse_RIGHT(dx, dy)
            return

    def on_key_press(self, symbol, modifiers):
        key = pyglet.window.key
        if symbol == key.ESCAPE:
            pyglet.app.event_loop.exit()
        if symbol == key.R:
            x = self.camera.x
            y = self.camera.y
            z = self.camera.z
            rx = self.camera.rx
            ry = self.camera.ry
            rz = self.camera.rz
            for t in np.arange(0., 1., 0.1):
                t = 3 * t**2 - 2 * t**3
                self.camera.x = self.camera_init_setting["x"] * t + x * (1 - t)
                self.camera.y = self.camera_init_setting["y"] * t + y * (1 - t)
                self.camera.z = self.camera_init_setting["z"] * t + z * (1 - t)
                self.camera.rx = self.camera_init_setting["rx"] * t + rx * (1 - t)
                self.camera.ry = self.camera_init_setting["ry"] * t + ry * (1 - t)
                self.camera.rz = self.camera_init_setting["rz"] * t + rz * (1 - t)
                self.on_draw()
                self.flip()
            self.camera.__init__(self.camera_init_setting)
        if symbol == key.F8:
            rx = self.camera.rx
            rz = self.camera.rz
            closest_rx = round(rx / 90) * 90
            closest_rz = round(rz / 90) * 90
            for t in np.arange(0., 1., 0.1):
                t = 3 * t**2 - 2 * t**3
                self.camera.rx = closest_rx * t + rx * (1 - t)
                self.camera.rz = closest_rz * t + rz * (1 - t)
                self.on_draw()
                self.flip()
            self.camera.rx = closest_rx
            self.camera.rz = closest_rz
        if symbol == key.Q:
            # print(self.camera.rx, self.camera.ry, self.camera.rz)
            # print(self.camera.x, self.camera.y, self.camera.z)
            self.mouse_detect_ON = not self.mouse_detect_ON

    # def on_window_close(self, window):
    #     event_loop.exit()

class GL_camera(Window):
    def __init__(self, setting = None):
        if setting:
            self.x = setting["x"]
            self.y = setting["y"]
            self.z = setting["z"]
            self.rx = setting["rx"]
            self.ry = setting["ry"]
            self.rz = setting["rz"]
        else:
            self.x = self.y = 0
            self.z = -10
            self.rx = self.ry = self.rz = 0
    def on_mouse_scroll(self, scroll_x, scroll_y): # zoom
        with np.errstate(divide = 'ignore', over = 'ignore'):
            self.z = min(-0.5, self.z + scroll_y * 10)
            # * (1 + 1 / (1 + np.exp((self.z + 5) * 5)**(-1)))
    # def MOD_CTRL_mouse_MIDDLE(self, angle):
    #     self.rz += angle
    def mouse_RIGHT(self, dx, dy):
        self.x += dx / 10.0
        self.y += dy / 10.0
    def mouse_MIDDLE(self, dx, dy):
        self.rz += dx / 5.0
        self.rx -= dy / 5.0

class Surface(Window):
    def __init__(self, window, mx, my, mz, color):
        # super().__init__()
        self.window = window
        vertices = []
        self.w, self.h = mz.shape
        # indices = []
        # r, c = mz.shape
        # pts = np.column_stack([mx.reshape(-1), my.reshape(-1), mz.reshape(-1)])

        for j in range(self.h - 1):
            # a row of triangles
            row = []
            for i in range(self.w):
                row.extend((mx[i, j], my[i, j], mz[i, j]))
                row.extend((mx[i, j + 1], my[i, j + 1], mz[i, j + 1]))
            vertices.append(row)

        colormax = np.array(vertices).reshape(-1, 3)[:, 2]
        colormin = np.amin(colormax)
        colormax = np.amax(colormax)

        self.rows = []
        self.colors = []
        for row in vertices:
            row = np.array(row).reshape(-1, 3)
            color_= 1 - (row[:, 2] - colormin) / (colormax - colormin) if colormax != colormin else np.zeros_like(row[:, 2])
            if color == 'b':
                DC = np.array([0, 100, 255, 0.7 * 255]) / 255
                LC = np.array([100, 180, 255, 0.7 * 255]) / 255
                # color_= np.column_stack([color_, color_, np.ones_like(color_), np.zeros_like(color_) + 0.7]).reshape(-1)
            if color == 'g':
                DC = np.array([0, 153, 51, 0.7 * 255]) / 255
                LC = np.array([153, 255, 204, 0.7 * 255]) / 255
                # color_ = np.column_stack([color_, np.ones_like(color_), color_, np.zeros_like(color_) + 0.7]).reshape(-1)
            if color == 'r':
                DC = np.array([204, 0, 0, 0.7 * 255]) / 255
                LC = np.array([255, 153, 204, 0.7 * 255]) / 255
                # color_ = np.column_stack([np.ones_like(color_), color_, color_, np.zeros_like(color_) + 0.7]).reshape(-1)
            DC = np.tile(DC, (len(color_), 1))
            LC = np.tile(LC, (len(color_), 1))
            color_= color_[..., None]
            color_= (LC * (1 - color_) + DC * color_).reshape(-1)
            self.colors.append(color_)
            self.rows.append(row.reshape(-1))
        # self.colors = np.array(self.colors).reshape(-1)
        # self.rows = np.array(self.rows).reshape(-1)

    def draw(self):
        glLoadIdentity()
        glTranslatef(self.window.camera.x, self.window.camera.y, self.window.camera.z)
        glRotatef(self.window.camera.rx, 1, 0, 0)
        glRotatef(self.window.camera.ry, 0, 1, 0)
        glRotatef(self.window.camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        for i in range(len(self.rows)):
            pyglet.graphics.draw(self.w * 2, GL_TRIANGLE_STRIP, 
                ('v3f', self.rows[i]), 
                ('c4f', self.colors[i]))

        # self.batch.draw()

class Line(Window):
    def __init__(self, window, line_points, line_width):
        # super().__init__()
        self.window = window
        self.line_count = len(line_points)
        self.line_points = line_points
        # self.line_points[:, 1] *= -1
        self.line_colors = np.column_stack([np.full_like(self.line_points, 0.5), np.ones(len(self.line_points))]).reshape(-1)
        self.line_width = line_width
        self.line_points = self.line_points.reshape(-1)

    def draw(self):
        glLoadIdentity()
        glTranslatef(self.window.camera.x, self.window.camera.y, self.window.camera.z)
        glRotatef(self.window.camera.rx, 1, 0, 0)
        glRotatef(self.window.camera.ry, 0, 1, 0)
        glRotatef(self.window.camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        glLineWidth(self.line_width)
        pyglet.graphics.draw(self.line_count, GL_LINES,
            ('v3f', self.line_points), 
            ('c4f', self.line_colors))

class Point(Window):
    def __init__(self, window, points, pointsize, color = None):
        # super().__init__()
        self.window = window
        self.points_count = len(points)
        self.points = np.array(points)
        # self.points[:, 1] *= -1
        self.pointsize = pointsize
        if color is None:
            self.data_points_color = self.color_plot()
            # self.data_points_color = np.column_stack([self.data_points_color, np.ones(len(self.data_points_color))]).reshape(-1)
            self.data_points_color = self.data_points_color.reshape(-1)
        else:
            if color == "k":
                self.data_points_color = np.tile(np.zeros(3), (len(self.points), 1)).reshape(-1)
            if color == "r":
                self.data_points_color = np.tile(np.array([1., 0., 0.]), (len(self.points), 1)).reshape(-1)
        self.points = self.points.reshape(-1)

    def color_plot(self):
        h = self.points[:, 2]
        if np.max(h) == np.min(h):
            return np.column_stack([np.ones_like(h)] * 3)
        h = (h - np.min(h)) / (np.max(h) - np.min(h))
        h = h * 0.8 + 0.2
        return np.array(list(map(lambda x: hsv_to_rgb(x, 1., 1.), h))) * 0.75

    def draw(self):
        glLoadIdentity()
        glTranslatef(self.window.camera.x, self.window.camera.y, self.window.camera.z)
        glRotatef(self.window.camera.rx, 1, 0, 0)
        glRotatef(self.window.camera.ry, 0, 1, 0)
        glRotatef(self.window.camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        glPointSize(self.pointsize)
        pyglet.graphics.draw(self.points_count, GL_POINTS,
            ('v3f', self.points),
            ('c3f', self.data_points_color))

if __name__ == "__main__":
    user32 = ctypes.windll.user32
    screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    width = int(screensize[0] * 0.8)
    height = int(screensize[1] * 0.8)
    window = Window(width = width, height = height, caption = 'GL_Plot', resizable=True)
    pyglet.app.run()
