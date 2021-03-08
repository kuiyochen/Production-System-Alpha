# http://www.poketcode.com/en/pyglet_demos/index.html
import os
import numpy as np
from colorsys import hsv_to_rgb

import pyglet
from pyglet.gl import *


class GL_camera():
    def __init__(self):
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
camera = GL_camera()

class Surface():
    def __init__(self, mx, my, mz, color):
        super().__init__()
        vertices = []
        self.width, self.height = mz.shape
        # indices = []
        # r, c = mz.shape
        # pts = np.column_stack([mx.reshape(-1), my.reshape(-1), mz.reshape(-1)])

        for j in range(self.height - 1):
            # a row of triangles
            row = []
            for i in range(self.width):
                row.extend((mx[i, j], my[i, j], mz[i, j]))
                row.extend((mx[i, j + 1], my[i, j + 1], mz[i, j + 1]))
            vertices.append(row)

        # # for j in range(c - 1):
        # #     # row = []
        # #     # for i in range(r):
        # #     #     row.append(c * i + j)
        # #     #     row.append(row[-1] + 1)
        # #     temp = np.tile(np.arange(r) * c + j, (1, 2))
        # #     temp[:, 1] += 1
        # #     indices.append(temp.reshape(-1).astype(int))
        # temp = np.arange(r * c).reshape(r, c)
        # temp = np.concatenate([temp[:-1, :][..., None], temp[1:, :][..., None]], axis = -1)
        # indices = temp.reshape(r - 1, -1).astype(int)

        # self.batch = pyglet.graphics.Batch()
        # for i in range(r - 1):
        #     if i>0:
        #         break
        #     self.batch.add_indexed(len(indices[i]),
        #                             GL_TRIANGLE_STRIP,
        #                             pyglet.graphics.Group(),
        #                             indices[i],
        #                             ('v3f/static', pts.reshape(-1)),
        #                             ('c4f/static', [0., 1., 0.5, 0.7] * len(indices[i])))

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
        global camera
        glTranslatef(camera.x, camera.y, camera.z)
        glRotatef(camera.rx, 1, 0, 0)
        glRotatef(camera.ry, 0, 1, 0)
        glRotatef(camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        for i in range(len(self.rows)):
            pyglet.graphics.draw(self.width * 2, GL_TRIANGLE_STRIP, 
                ('v3f', self.rows[i]), 
                ('c4f', self.colors[i]))

        # self.batch.draw()

class Line():
    def __init__(self, line_points, line_width):
        super().__init__()
        self.line_count = len(line_points)
        self.line_points = line_points
        # self.line_points[:, 1] *= -1
        self.line_colors = np.column_stack([np.full_like(self.line_points, 0.5), np.ones(len(self.line_points))]).reshape(-1)
        self.line_width = line_width
        self.line_points = self.line_points.reshape(-1)

    def draw(self):
        glLoadIdentity()
        global camera
        glTranslatef(camera.x, camera.y, camera.z)
        glRotatef(camera.rx, 1, 0, 0)
        glRotatef(camera.ry, 0, 1, 0)
        glRotatef(camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        glLineWidth(self.line_width)
        pyglet.graphics.draw(self.line_count, GL_LINES,
            ('v3f', self.line_points), 
            ('c4f', self.line_colors))

class Point():
    def __init__(self, points, pointsize):
        super().__init__()
        self.points_count = len(points)
        self.points = np.array(points)
        # self.points[:, 1] *= -1
        self.pointsize = pointsize
        self.data_points_color = self.color_plot()
        # self.data_points_color = np.column_stack([self.data_points_color, np.ones(len(self.data_points_color))]).reshape(-1)
        self.data_points_color = self.data_points_color.reshape(-1)
        self.points = self.points.reshape(-1)

    def color_plot(self):
        h = self.points[:, 2]
        if np.max(h) == np.min(h):
            return np.column_stack([np.ones_like(h)] * 3)
        h = (h - np.min(h)) / (np.max(h) - np.min(h))
        h = h * 0.8 + 0.2
        return np.array(list(map(lambda x: hsv_to_rgb(x, 1., 1.), h)))

    def draw(self):
        glLoadIdentity()
        global camera
        glTranslatef(camera.x, camera.y, camera.z)
        glRotatef(camera.rx, 1, 0, 0)
        glRotatef(camera.ry, 0, 1, 0)
        glRotatef(camera.rz, 0, 0, 1)
        # glColor3f(0.5, 0.5, 0.5)

        glPointSize(self.pointsize)
        pyglet.graphics.draw(self.points_count, GL_POINTS,
            ('v3f', self.points),
            ('c3f', self.data_points_color))

width = 1000
height = 800

# config = pyglet.gl.Config(sample_buffers=1, samples=4)
# window = pyglet.window.Window(width=width, height=height, config=config, caption='GL_Plot', resizable=True)
window = pyglet.window.Window(width=width, height=height, caption='GL_Plot', resizable=True)
cursor = window.get_system_mouse_cursor(window.CURSOR_CROSSHAIR)
window.set_mouse_cursor(cursor)

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


GL_objs = []

# GL_objs.append(Point(np.zeros((1, 3)), 2))
GL_objs.append(Point(np.column_stack([np.arange(0, 10, 1.5), 
                        np.arange(0, 10, 1.5),
                        np.arange(0, 10, 1.5)])-5, 5))
# GL_objs.append(Line(np.array([[10.0, 10.0, 20], 
#                 [10.0, 0.0, -10], [30, 30, 30], [40, 40, 40]]), 2))
plot_grid = True
if plot_grid:
    temp = np.arange(-50, 50 + 1., 20)
    temp_ = np.column_stack([temp, np.full_like(temp, 50), np.zeros_like(temp)])
    temp__ = temp_.copy()
    temp__[:, 1] = -temp__[:, 1]
    temp_ = np.transpose(np.concatenate([temp_[..., None], temp__[..., None]], axis = -1), (0, 2, 1))
    GL_objs.append(Line(temp_.reshape(-1, 3), 1))

    temp_ = np.column_stack([np.full_like(temp, 50), temp, np.zeros_like(temp)])
    temp__ = temp_.copy()
    temp__[:, 0] = -temp__[:, 0]
    temp_ = np.transpose(np.concatenate([temp_[..., None], temp__[..., None]], axis = -1), (0, 2, 1))
    GL_objs.append(Line(temp_.reshape(-1, 3), 1))

    temp = np.array([10., 10, 20])
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
    GL_objs.append(Line(temp_.copy(), 1))

    GL_objs.append(Line(np.array([[0, 0, 0.], [100, 0, 0], \
                                [0, 0, 0.], [0, 100, 0], \
                                [0, 0, 0.], [0, 0, 20]]), 3))

x = np.arange(-30, 50, 1)
y = np.arange(-10, 100, 1)
mx, my = np.meshgrid(x, y, indexing='ij')
# mz = (mx**2 +my**2) / (5 * 10**2) + 10
mz = (mx**2 + my**2 - 20 * mx - 40 * my) / (5 * 10**2)
GL_objs.append(Surface(mx, my, mz, 'b'))

x = np.arange(-60, 60, 1)
y = np.arange(-60, 60, 1)
mx1, my1 = np.meshgrid(x, y, indexing='ij')
# mz1 = ((mx1 + 30)**2 + my1**2) / (10**3) + 15
mz1 = np.zeros_like(mx1)
GL_objs.append(Surface(mx1, my1, mz1, 'g'))

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45., width / height, 0.1, 800.)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    return pyglet.event.EVENT_HANDLED

@window.event
def on_draw():

    # clears the background with the background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # wire-frame mode
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    # glColor3f(0.5, 0.5, 0.5)

    # glBegin(GL_POINTS)
    # glBegin(GL_LINES)
    # glColor4fv((GLfloat * 4)(1, 0, 0, 1))
    # glVertex3f(10.0, 10.0, 20)
    # glVertex3f(0.0, 0.0, 20)
    # glEnd()
    # lines.draw()

    for obj in GL_objs:
        obj.draw()

    glFlush()

# @window.event
# def on_mouse_motion(x, y, dx, dy):
#     # window.clear()
#     # print(x, y)
#     print("-------------")
#     thetas = np.array([camera.rx, camera.ry, camera.rz])
#     cx, cy, cz = np.cos(np.radians(thetas))
#     sx, sy, sz = np.sin(np.radians(thetas))
#     Rotate_matrix = np.dot(np.array([[cz, sz, 0.], 
#                                     [-sz, cz, 0.], 
#                                     [0., 0., 1.]]), 
#                     np.dot(np.array([[cy, 0., -sy], 
#                                     [0., 1., 0.], 
#                                     [sy, 0., cy]]), 
#                             np.array([[1., 0., 0.], 
#                                     [0., cx, sx], 
#                                     [0., -sx, cx]])))
#     camera_pos = -np.array([camera.x, camera.y, camera.z])
#     camera_pos = np.dot(Rotate_matrix, camera_pos[..., None]).reshape(-1)
#     with np.errstate(divide='ignore'):
#         camera_pos = camera_pos / np.linalg.norm(camera_pos)
#         camera_pos = np.nan_to_num(camera_pos, nan = 0., posinf = 0., neginf = 0.)
#     GL_objs[0] = Point(np.array([camera_pos / i for i in range(1, 10)]) * 2, 5)
#     print(thetas)
#     print(camera_pos)
#     # GL_objs[1] = Point(np.zeros(3)[None, ...] / 10 - 0.5, 20)
#     # GL_objs[1] = Line(np.array([camera_pos, np.zeros(3)]), 2)
#     # GL_objs.insert(1, )
#     # del GL_objs[1]


@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y): # zoom
    camera.on_mouse_scroll(scroll_x, scroll_y)

@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    key = pyglet.window.key
    # if button == pyglet.window.mouse.MIDDLE and (modifiers & key.MOD_CTRL):
    #     x -= window.width // 2
    #     y -= window.height // 2
    #     lx = x - dx
    #     ly = y - dy
    #     angle = np.angle((x + y * 1J) / ((lx + ly * 1J))) * 50
    #     camera.MOD_CTRL_mouse_MIDDLE(angle)
    #     return
    if button == pyglet.window.mouse.MIDDLE:
        camera.mouse_MIDDLE(dx, dy)
        return
    if button == pyglet.window.mouse.RIGHT:
        camera.mouse_RIGHT(dx, dy)
        return

@window.event
def on_key_press(symbol, modifiers):
    key = pyglet.window.key
    if symbol == key.R:
        camera.__init__()

pyglet.app.run()
