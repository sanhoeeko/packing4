from math import sin, cos

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from .kernel_for_test import ker


class XytPair:
    def __init__(self, lst):
        assert len(lst) == 6
        self.f1 = lst[0:2]
        self.m1 = lst[2]
        self.f2 = lst[3:5]
        self.m2 = lst[5]

    @property
    def first(self):
        return self.f1, self.m1

    @property
    def second(self):
        return self.f2, self.m2


class Shape:
    def __init__(self, n, d):
        ker.setEnums(1)
        ker.setRod(n, d)
        self.func = ker.gradientTest

    def reference(self):
        return ker.gradientReference


class ForceInterface:
    def __init__(self, n, d):
        self.shape = Shape(n, d)
        self.ref = self.shape.reference()

    def calForce(self, dx, dy, t1, t2):
        xyt_pair = XytPair(self.shape.func(dx, dy, t1, t2))
        return xyt_pair

    def calForceReference(self, dx, dy, t1, t2):
        xyt_pair = XytPair(self.ref.func(dx, dy, t1, t2))
        return xyt_pair

    def calPoint(self, theta, center, F, moment):
        u = np.array([cos(theta), sin(theta)])
        r = moment / (np.cross(u, F))
        force_act_on_point = center + r * u
        return force_act_on_point


def drawPlacedVector(vector: np.ndarray, pos: np.ndarray):
    # print(f"Draw force: ({pos[0]}, {pos[1]}), ({vector[0]}, {vector[1]})")
    ax.arrow(pos[0], pos[1], vector[0] * 0.1, vector[1] * 0.1, head_width=0.1, head_length=0.1)


class DraggablePoint:
    lock = None

    def __init__(self, point, line, enable_force=True, n=2, d=2):
        self.enable_force = enable_force
        self.point = point
        self.line = line
        self.theta = 0
        self.press = None
        self.background = None
        xp, yp, tp = self.calMirror()
        self.mirror_point = plt.Circle((xp, yp), 0.2, color='orange')
        self.mirror_line, = ax.plot((xp, yp), (px + np.cos(tp), py + np.sin(tp)), 'r-', color='blue')
        ax.add_patch(self.point)
        ax.add_patch(self.mirror_point)
        self.force_interface = None
        if self.enable_force:
            self.force_interface = ForceInterface(n, d)

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidscroll = self.point.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

    def clearCanvas(self):
        def isNotIn(obj, lst):
            for x in lst:
                if obj is x: return False
            return True

        for child in plt.gca().get_children():
            if isNotIn(child, [self.point, self.line, self.mirror_point, self.mirror_line, ellipse]):
                try:
                    child.remove()
                except NotImplementedError:
                    pass

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        axes.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self: return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0] + dx, self.point.center[1] + dy)

        # update self.line
        x = np.array([-100, 100])  # large enough to cover the plot area
        y = np.tan(self.theta) * (x - self.point.center[0]) + self.point.center[1]
        self.line.set_data(x, y)

        # update mirror point and line
        xp, yp, tp = self.calMirror()
        self.mirror_point.center = (xp, yp)
        y_mirror = np.tan(tp) * (x - xp) + yp
        self.mirror_line.set_data(x, y_mirror)

        canvas = self.point.figure.canvas
        axes = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        axes.draw_artist(self.point)
        axes.draw_artist(self.line)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self: return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def on_scroll(self, event):
        'rotate self.line on scroll'
        if event.inaxes != self.point.axes: return
        self.theta += np.deg2rad(5) if event.button == 'up' else np.deg2rad(-5)
        x = np.array([-100, 100])  # large enough to cover the plot area
        y = np.tan(self.theta) * (x - self.point.center[0]) + self.point.center[1]
        self.line.set_data(x, y)

        # update mirror point and line
        xp, yp, tp = self.calMirror()
        self.mirror_point.center = (xp, yp)
        y_mirror = np.tan(tp) * (x - xp) + yp
        self.mirror_line.set_data(x, y_mirror)

        # plot force
        if self.enable_force:
            self.clearCanvas()
            force1, point1, force2, point2 = self.calForce('test')
            if np.linalg.norm(force1) > 0:
                drawPlacedVector(force1, point1)
            if np.linalg.norm(force2) > 0:
                drawPlacedVector(force2, point2)

        self.line.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)
        self.point.figure.canvas.mpl_disconnect(self.cidscroll)

    def getXyt(self):
        'get the x, y position and theta (rotation angle) of the point'
        return self.point.center[0], self.point.center[1], self.theta

    def calMirror(self):
        'invoke the kernel to calculate the mirror point / orientation'
        x, y, t = self.getXyt()
        xp, yp, tp = ker.getMirrorOf(A, B, x, y, t)
        if xp == 0 and yp == 0 and tp == 0:
            if not self.enable_force:
                print("Invalid collision")
            return np.nan, np.nan, np.nan
        else:
            if not self.enable_force:
                print(xp, yp, tp)
            return xp, yp, tp

    def calMirrorXYTT(self):
        x, y, t = self.getXyt()
        xp, yp, tp = self.calMirror()
        return x - xp, y - yp, t, tp

    def calForce(self, mode='test'):
        dx, dy, t1, t2 = self.calMirrorXYTT()

        if mode == 'test':
            xyt_pair = self.force_interface.calForce(dx, dy, t1, t2)
        elif mode == 'ref':
            xyt_pair = self.force_interface.calForceReference(dx, dy, t1, t2)

        F1, M1 = xyt_pair.first
        c1 = self.getXyt()[:2]
        P1 = self.force_interface.calPoint(t1, c1, F1, M1)
        F2, M2 = xyt_pair.second
        c2 = self.calMirror()[:2]
        P2 = self.force_interface.calPoint(t2, c2, F2, M2)
        print(f"F1={F1}, F2={F2}, M1={M1}, M2={M2}")
        return F1, P1, F2, P2


fig, ax = plt.subplots()

# draw the boundary
A, B = 12, 24
y0, y1 = -26, -18
x0, x1 = -6, 2
ellipse = patches.Ellipse((0, 0), width=2 * A, height=2 * B, fill=False)
ax.add_artist(ellipse)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect('equal')

# draw a point and a line
px, py = (x0 + x1) / 2, (y0 + y1) / 2
point = plt.Circle((px, py), 0.2)
line, = ax.plot((px, py), (px + 1, py + 1), 'r-')

draggable_point = DraggablePoint(point, line)
draggable_point.connect()

plt.show()
